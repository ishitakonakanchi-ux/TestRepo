"""
Example: load a trained NPE and inspect posteriors.

Usage:
    python example_transit.py weights/npe_2026-04-14_15h30m.pkl

Requires running train_sbi.py first.
"""

import os
import sys
import pickle
import logging
import warnings

logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="sbi")
warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")

import numpy as np
import torch
import matplotlib.pyplot as plt
import emcee
import corner
from tqdm import trange
import jax.numpy as jnp
from jax import jit

from npe_wrapper import NPEEstimator

SEED = 42
from transit_sbi import (
    simulate_dataset, simulator, t_obs,
    PRIOR_LOW, PRIOR_HIGH, PARAM_LABELS, SIGMA, N_OBS,
)

SHOW_FIG = False
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Parse arguments ──────────────────────────────────────────────────────
if len(sys.argv) != 2:
    print("Usage: python example_transit.py <weights_file>")
    print("Example: python example_transit.py weights/npe_2026-04-14_15h30m.pkl")
    sys.exit(1)

model_fname = sys.argv[1]
summary_fname = model_fname.replace("npe_", "summary_")

if not os.path.exists(model_fname):
    raise FileNotFoundError(f"Weights file not found: {model_fname}")

# ── Load trained model ───────────────────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Loading {model_fname}")
npe = NPEEstimator().load(model_fname)
with open(summary_fname, "rb") as f:
    summaries = pickle.load(f)

# ── 1. Training loss vs epoch ────────────────────────────────────────────
summary = summaries[0]
fig, ax = plt.subplots()
ax.plot(summary["training_loss"], label="Training")
ax.plot(summary["validation_loss"], label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fname = os.path.join(PLOT_DIR, "training_loss.png")
print(f"Saving {fname}")
fig.savefig(fname, dpi=150, bbox_inches="tight")

# ── 2. Two synthetic observations at different true parameters ───────────
true_A = np.array([0.35, 0.15, 0.10])  # low impact, short duration
true_B = np.array([0.55, 0.25, 0.17])  # moderate impact, longer duration

x_obs_A = np.array(simulator(true_A)) + np.random.normal(0, SIGMA, N_OBS)
x_obs_B = np.array(simulator(true_B)) + np.random.normal(0, SIGMA, N_OBS)

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

# ── 3. Corner plot: two observations ─────────────────────────────────────
fig = corner.corner(samples_A, labels=PARAM_LABELS, truths=true_A,
                    color="C0", hist_kwargs={"density": True})
corner.corner(samples_B, fig=fig, truths=true_B,
              color="C1", hist_kwargs={"density": True})
fig.legend([plt.Line2D([], [], color="C0"), plt.Line2D([], [], color="C1")],
           ["Observation A", "Observation B"], loc="upper right", fontsize=12)
fname = os.path.join(PLOT_DIR, "posterior_corner.png")
print(f"Saving {fname}")
fig.savefig(fname, dpi=150, bbox_inches="tight")

# ── 4. MCMC comparison ───────────────────────────────────────────────────
@jit
def log_prior_MCMC(theta):
    in_bounds = jnp.all((jnp.array(PRIOR_LOW) <= theta) &
                        (theta <= jnp.array(PRIOR_HIGH)))
    return jnp.where(in_bounds, 0.0, -jnp.inf)

@jit
def log_likelihood_MCMC(theta):
    model = simulator(theta)
    return -0.5 * jnp.sum(((x_obs_A - model) / SIGMA)**2)

@jit
def log_posterior_MCMC(theta):
    lp = log_prior_MCMC(theta)
    return jnp.where(jnp.isfinite(lp), lp + log_likelihood_MCMC(theta), -jnp.inf)

ndim, nwalkers, nsteps, nburn = 3, 32, 5000, 1000
p0 = true_A + 1e-3 * np.random.normal(size=(nwalkers, ndim))
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_MCMC)
sampler.run_mcmc(p0, nsteps, progress=True)
flat_samples = sampler.get_chain(discard=nburn, flat=True)

print("\nMCMC posterior for observation A:")
for i, label in enumerate(PARAM_LABELS):
    print(f"  {label}: true={true_A[i]:.4f}, "
          f"mean={flat_samples[:, i].mean():.4f} +/- {flat_samples[:, i].std():.4f}")

fig = corner.corner(samples_A, labels=PARAM_LABELS,
                    truths=true_A, color="C0", hist_kwargs={"density": True})
corner.corner(flat_samples, fig=fig, color="C2", hist_kwargs={"density": True})
fig.legend([plt.Line2D([], [], color="C0"), plt.Line2D([], [], color="C2")],
           ["NPE", "MCMC"], loc="upper right", fontsize=12)
fname = os.path.join(PLOT_DIR, "posterior_sbi_vs_mcmc.png")
print(f"Saving {fname}")
fig.savefig(fname, dpi=150, bbox_inches="tight")

# ── 5. Posterior predictive light curves ─────────────────────────────────
t_np = np.array(t_obs)
fig, ax = plt.subplots()
for i in range(50):
    ax.plot(t_np, simulator(samples_A[i]), color="C0", alpha=0.05)
ax.plot(t_np, x_obs_A, "k.", ms=2, label="Observed")
ax.set_xlabel(r"$t - t_0$ [days]")
ax.set_ylabel("Relative flux")
ax.legend()
fname = os.path.join(PLOT_DIR, "posterior_predictive.png")
print(f"Saving {fname}")
fig.savefig(fname, dpi=150, bbox_inches="tight")

# ── 6. PIT calibration ───────────────────────────────────────────────────
n_test = 200
theta_test, x_test = simulate_dataset(n_test)
pit_values = np.empty((n_test, len(PARAM_LABELS)))

for i in trange(n_test, desc="PIT"):
    samples_i = npe.sample(x_test[i], n_samples=2000, show_progress_bars=False)
    pit_values[i] = np.mean(samples_i < theta_test[i], axis=0)

fig, axes = plt.subplots(1, len(PARAM_LABELS), figsize=(12, 3))
for j, (ax, label) in enumerate(zip(axes, PARAM_LABELS)):
    ax.hist(pit_values[:, j], bins=20, density=True, edgecolor="k", alpha=0.7)
    ax.axhline(1.0, color="k", ls="--")
    ax.set_xlabel(f"PIT({label})")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
fname = os.path.join(PLOT_DIR, "pit_calibration.png")
print(f"Saving {fname}")
fig.savefig(fname, dpi=150, bbox_inches="tight")

if SHOW_FIG:
    plt.show()
