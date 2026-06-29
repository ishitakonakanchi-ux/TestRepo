"""
Example: load a trained NPE and inspect posteriors.

Usage:
    python example_transit.py [weights_file]

If no weights file is specified, uses the most recent model in weights/.
Requires running train_sbi.py first.
"""

import glob
import logging
import os
import sys
import warnings

logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="sbi")
warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")

MPLCONFIGDIR = os.path.join("data", "matplotlib-cache")
PYTHON_CACHE_DIR = os.path.join("data", "python-cache")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.makedirs(PYTHON_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", os.path.abspath(PYTHON_CACHE_DIR))
home_cache = os.path.expanduser("~/Library/Caches")
if not os.access(home_cache, os.W_OK):
    local_home = os.path.abspath(os.path.join("data", "python-home"))
    os.makedirs(os.path.join(local_home, "Library", "Caches"), exist_ok=True)
    os.environ["HOME"] = local_home

import numpy as np
import torch
import matplotlib.pyplot as plt
import corner
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

from npe_wrapper import NPEEstimator
from train_sbi import CNNEmbedding  # noqa: F401 (needed for pickle)

SEED = 42
from transit_sbi import (
    simulator, t_obs, PRIOR_LOW, PRIOR_HIGH, PARAM_LABELS, SIGMA, N_OBS,
)

SHOW_FIG = False
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

#loading flux error(KIC 008112013) 
TARGET_OBJ_IDX = 0
_lib = np.load(
    "data/dr25_dv_library/dr25_dv_sbi_library.npz", allow_pickle=True)
TARGET_FLUX_ERR = np.asarray(
    _lib["flux_err"][TARGET_OBJ_IDX], dtype=np.float32)
TARGET_FLUX_ERR_JAX = jnp.array(TARGET_FLUX_ERR)
print(f"Using per-point flux_err from "
      f"{_lib['name'][TARGET_OBJ_IDX]}")
print(f"  flux_err median: {np.median(TARGET_FLUX_ERR):.2e}")

# ── Parse arguments ──────────────────────────────────────────────────────
if len(sys.argv) == 2:
    model_fname = sys.argv[1]
else:
    # preferring flux error model, otherwise take latest model
    candidates = glob.glob("weights/npe_fluxerr_*.pkl")
    if not candidates:
        candidates = glob.glob("weights/npe_*.pkl")
    if not candidates:
        print("No model files found in weights/. Run train_sbi.py first.")
        sys.exit(1)
    model_fname = max(candidates, key=os.path.getmtime)
    print(f"Using latest model: {model_fname}")

if not os.path.exists(model_fname):
    raise FileNotFoundError(f"Weights file not found: {model_fname}")


# ── Load trained model ───────────────────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Loading {model_fname}")

npe = NPEEstimator().load(model_fname)

# ── 1. Generate two synthetic observations ───────────────────────────────
true_A = np.array([0.35, 0.15, 0.10, 0.3, 0.2, 2e-3])  # low impact, short duration
true_B = np.array([0.55, 0.25, 0.17, 0.4, 0.3, 2e-3])  # moderate impact, longer duration

#using real noise
x_obs_A = np.array(simulator(true_A)) + np.random.normal(0, TARGET_FLUX_ERR, N_OBS) 
x_obs_B = np.array(simulator(true_B)) + np.random.normal(0, TARGET_FLUX_ERR, N_OBS)

samples_A = npe.sample(x_obs_A, n_samples=10_000)

samples_B = npe.sample(x_obs_B, n_samples=10_000)

print("\nSBI posterior for observation A:")
for i, label in enumerate(PARAM_LABELS):
    print(f"  {label}: true={true_A[i]:.4f}, "
          f"mean={samples_A[:, i].mean():.4f} +/- {samples_A[:, i].std():.4f}")
print("\nSBI posterior for observation B:")
for i, label in enumerate(PARAM_LABELS):
    print(f"  {label}: true={true_B[i]:.4f}, "
          f"mean={samples_B[:, i].mean():.4f} +/- {samples_B[:, i].std():.4f}")

# Contour levels: 1-sigma and 2-sigma (68% and 95%)
LEVELS = (1 - np.exp(-0.5 * np.array([1, 2])**2))

# ── 2. Corner plot: two observations ─────────────────────────────────────
fig = corner.corner(samples_A, labels=PARAM_LABELS, truths=true_A,
                    color="C0", truth_color="red", smooth=1.0, levels=LEVELS,
                    hist_kwargs={"density": True})
corner.corner(samples_B, fig=fig, truths=true_B,
              color="C1", truth_color="red", smooth=1.0, levels=LEVELS,
              hist_kwargs={"density": True})
fig.legend([plt.Line2D([], [], color="C0"), plt.Line2D([], [], color="C1")],
           ["Observation A", "Observation B"], loc="upper right", fontsize=12)
fname = os.path.join(PLOT_DIR, "posterior_corner_fluxerr.png") # renamed
print(f"Saving {fname}")
fig.savefig(fname, dpi=150, bbox_inches="tight")

# ── 3. MCMC comparison for both observations ─────────────────────────────
MCMC_NUM_WARMUP = 1000
MCMC_NUM_SAMPLES = 1000
MCMC_NUM_CHAINS = 2
MCMC_TARGET_ACCEPT = 0.9


def transit_numpyro_model(x_obs, t_grid, noise_array):
    theta = []
    for label, low, high in zip(PARAM_LABELS, PRIOR_LOW, PRIOR_HIGH):
        theta.append(numpyro.sample(label, dist.Uniform(low, high)))
    theta = jnp.stack(theta)
    model = simulator(theta, t_grid)
    numpyro.sample("obs", dist.Normal(model, noise_array).to_event(1), obs=x_obs)


def run_numpyro_mcmc(name, x_obs, t_grid=None, noise_array=None,
                     init_theta=None, seed_offset=0):
    if noise_array is None:
        noise_array = TARGET_FLUX_ERR
    if t_grid is None:
        t_grid = np.asarray(t_obs)
    if init_theta is None:
        init_theta = 0.5 * (np.asarray(PRIOR_LOW) + np.asarray(PRIOR_HIGH))

    prior_low = np.asarray(PRIOR_LOW)
    prior_high = np.asarray(PRIOR_HIGH)
    init_theta = np.clip(init_theta, prior_low + 1e-6, prior_high - 1e-6)
    init_values = {
        label: float(value) for label, value in zip(PARAM_LABELS, init_theta)
    }

    print(f"\nRunning NumPyro MCMC for {name}...")
    kernel = NUTS(
        transit_numpyro_model,
        dense_mass=True,
        target_accept_prob=MCMC_TARGET_ACCEPT,
        init_strategy=init_to_value(values=init_values),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=MCMC_NUM_WARMUP,
        num_samples=MCMC_NUM_SAMPLES,
        num_chains=MCMC_NUM_CHAINS,
        chain_method="vectorized",
        progress_bar=True,
    )
    mcmc.run(
        jax.random.PRNGKey(SEED + seed_offset),
        x_obs=jnp.asarray(x_obs),
        t_grid=jnp.asarray(t_grid),
        noise_array=jnp.asarray(noise_array),
        extra_fields=("diverging", "accept_prob", "num_steps"),
    )
    mcmc.print_summary()

    extra_fields = mcmc.get_extra_fields()
    if "diverging" in extra_fields:
        diverging = np.asarray(extra_fields["diverging"], dtype=bool)
        print(f"Divergences: {int(np.sum(diverging))} / {diverging.size}")
    if "accept_prob" in extra_fields:
        accept_prob = np.asarray(extra_fields["accept_prob"], dtype=float)
        print(f"Mean accept_prob: {np.nanmean(accept_prob):.3f}")

    samples = mcmc.get_samples()
    return np.column_stack([np.asarray(samples[label]) for label in PARAM_LABELS])


mcmc_samples_A = run_numpyro_mcmc(
    "observation A", x_obs_A, init_theta=true_A, seed_offset=1)

print("\nMCMC posterior for observation A:")
for i, label in enumerate(PARAM_LABELS):
    print(f"  {label}: true={true_A[i]:.4f}, "
          f"mean={mcmc_samples_A[:, i].mean():.4f} +/- {mcmc_samples_A[:, i].std():.4f}")

mcmc_samples_B = run_numpyro_mcmc(
    "observation B", x_obs_B, init_theta=true_B, seed_offset=2)

print("\nMCMC posterior for observation B:")
for i, label in enumerate(PARAM_LABELS):
    print(f"  {label}: true={true_B[i]:.4f}, "
          f"mean={mcmc_samples_B[:, i].mean():.4f} +/- {mcmc_samples_B[:, i].std():.4f}")

# Corner plot: SBI vs MCMC for observation A
fig = corner.corner(samples_A, labels=PARAM_LABELS, truths=true_A,
                    color="C0", truth_color="red", smooth=1.0, levels=LEVELS,
                    hist_kwargs={"density": True})
corner.corner(mcmc_samples_A, fig=fig, smooth=1.0, levels=LEVELS,
              color="C2", hist_kwargs={"density": True})
fig.legend([plt.Line2D([], [], color="C0"), plt.Line2D([], [], color="C2")],
           ["NPE", "MCMC"], loc="upper right", fontsize=12)
fname = os.path.join(PLOT_DIR, "posterior_sbi_vs_mcmc_A_fluxerr.png") #renamed
print(f"Saving {fname}")
fig.savefig(fname, dpi=150, bbox_inches="tight")

# Corner plot: SBI vs MCMC for observation B
fig = corner.corner(samples_B, labels=PARAM_LABELS, truths=true_B,
                    color="C0", truth_color="red", smooth=1.0, levels=LEVELS,
                    hist_kwargs={"density": True})
corner.corner(mcmc_samples_B, fig=fig, smooth=1.0, levels=LEVELS,
              color="C2", hist_kwargs={"density": True})
fig.legend([plt.Line2D([], [], color="C0"), plt.Line2D([], [], color="C2")],
           ["NPE", "MCMC"], loc="upper right", fontsize=12)
fname = os.path.join(PLOT_DIR, "posterior_sbi_vs_mcmc_B_fluxerr.png") #renamed
print(f"Saving {fname}")
fig.savefig(fname, dpi=150, bbox_inches="tight")

# ── 4. Posterior predictive light curves ─────────────────────────────────
t_np = np.array(t_obs)
fig, ax = plt.subplots()
for i in range(50):
    ax.plot(t_np, simulator(samples_A[i]), color="C0", alpha=0.05)
ax.plot(t_np, simulator(true_A), "r-", lw=2, label="Truth")
ax.plot(t_np, x_obs_A, "k.", ms=2, label="Observed")
ax.set_xlabel(r"$t - t_0$ [days]")
ax.set_ylabel("Relative flux")
ax.legend()
fname = os.path.join(PLOT_DIR, "posterior_predictive_fluxerr.png")#renamed
print(f"Saving {fname}")
fig.savefig(fname, dpi=150, bbox_inches="tight")

# ── 5. Kepler observation ───────────────────────────────────────────
print("\nLoading real Kepler data...")
library = np.load(
    "data/dr25_dv_library/dr25_dv_sbi_library.npz",
    allow_pickle=True)


index = 0
name      = library["name"][index]
t_kep     = jnp.array(library["phase_time"][index])
x_kep     = np.array(library["flux"][index])
flux_err_kep = np.array(library["flux_err"][index]) #pulling per-point flux 

print(f"Running inference on: {name}")

# NPE
samples_kep = npe.sample(x_kep, n_samples=10_000) #using real flux
print("\nNPE posterior for real Kepler planet:")
for i, label in enumerate(PARAM_LABELS):
    print(f"  {label}: {samples_kep[:,i].mean():.4f} "
          f"+/- {samples_kep[:,i].std():.4f}")

# MCMC
init_kep = np.array([
    0.6,
    float(library["dv_duration_hours"][index]) / 24.0,
    np.sqrt(float(library["dv_depth_ppm"][index]) * 1e-6),
    float(library["dv_period_days"][index]),
    0.0,
    0.3,
    0.2,
])
mcmc_samples_kep = run_numpyro_mcmc(
    "Kepler data", x_kep, t_kep, noise_array=flux_err_kep,
    init_theta=init_kep, seed_offset=3)

print("\nMCMC posterior for real Kepler planet:")
for i, label in enumerate(PARAM_LABELS): #mean and uncertainty of MCMC for Kepler data
    print(f"  {label}: {mcmc_samples_kep[:,i].mean():.4f} "
          f"+/- {mcmc_samples_kep[:,i].std():.4f}")

#  NPE vs MCMC corner plot
fig = corner.corner(samples_kep, labels=PARAM_LABELS,
                    color="C0", smooth=1.0, levels=LEVELS,
                    hist_kwargs={"density": True})
corner.corner(mcmc_samples_kep, fig=fig, smooth=1.0, levels=LEVELS,
              color="C2", hist_kwargs={"density": True})
fig.legend([plt.Line2D([], [], color="C0"),
            plt.Line2D([], [], color="C2")],
           ["NPE", "MCMC"], loc="upper right", fontsize=12)
fig.suptitle(f"NPE vs MCMC: {name}", fontsize=10)
fname = os.path.join(PLOT_DIR, "kepler_npe_vs_mcmc_fluxerr.png") #renamed
fig.savefig(fname, dpi=150, bbox_inches="tight")
print(f"Saved {fname}")

# posterior predictive plot
fig, ax = plt.subplots()
for i in range(50):
    flux_pred = np.array(simulator(samples_kep[i], t_kep))
    ax.plot(np.array(t_kep), flux_pred, color="C0", alpha=0.05)
ax.plot(np.array(t_kep), x_kep, "k.", ms=4, label="Kepler data")
ax.set_xlabel(r"$t - t_0$ [days]")
ax.set_ylabel("Relative flux")
ax.set_title(f"Posterior predictive: {name}")
ax.legend()
fname = os.path.join(PLOT_DIR, "kepler_posterior_predictive_fluxerr.png") # renamed
fig.savefig(fname, dpi=150, bbox_inches="tight")
print(f"Saved {fname}")

if SHOW_FIG:
    plt.show()
