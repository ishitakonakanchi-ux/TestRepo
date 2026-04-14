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

import numpy as np
import torch
import matplotlib.pyplot as plt
import emcee
import corner
import jax.numpy as jnp
from jax import jit

from npe_wrapper import NPEEstimator
from train_sbi import CNNEmbedding  # noqa: F401 (needed for pickle)

SEED = 42
from transit_sbi import (
    simulator, t_obs, PRIOR_LOW, PRIOR_HIGH, PARAM_LABELS, SIGMA, N_OBS,
)

SHOW_FIG = False
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Parse arguments ──────────────────────────────────────────────────────
if len(sys.argv) == 2:
    model_fname = sys.argv[1]
else:
    # Find the most recent model in weights/
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
fname = os.path.join(PLOT_DIR, "posterior_corner.png")
print(f"Saving {fname}")
fig.savefig(fname, dpi=150, bbox_inches="tight")

# ── 3. MCMC comparison for both observations ─────────────────────────────
@jit
def log_prior_MCMC(theta):
    in_bounds = jnp.all((jnp.array(PRIOR_LOW) <= theta) &
                        (theta <= jnp.array(PRIOR_HIGH)))
    return jnp.where(in_bounds, 0.0, -jnp.inf)


def make_log_posterior(x_obs):
    @jit
    def log_posterior(theta):
        lp = log_prior_MCMC(theta)
        model = simulator(theta)
        ll = -0.5 * jnp.sum(((x_obs - model) / SIGMA)**2)
        return jnp.where(jnp.isfinite(lp), lp + ll, -jnp.inf)
    return log_posterior


ndim, nwalkers, nsteps, nburn = 3, 32, 25000, 5000

# MCMC for observation A
print("\nRunning MCMC for observation A...")
log_post_A = make_log_posterior(x_obs_A)
p0_A = true_A + 1e-3 * np.random.normal(size=(nwalkers, ndim))
sampler_A = emcee.EnsembleSampler(nwalkers, ndim, log_post_A)
sampler_A.run_mcmc(p0_A, nsteps, progress=True)
mcmc_samples_A = sampler_A.get_chain(discard=nburn, thin=30, flat=True)

print("\nMCMC posterior for observation A:")
for i, label in enumerate(PARAM_LABELS):
    print(f"  {label}: true={true_A[i]:.4f}, "
          f"mean={mcmc_samples_A[:, i].mean():.4f} +/- {mcmc_samples_A[:, i].std():.4f}")

# MCMC for observation B
print("\nRunning MCMC for observation B...")
log_post_B = make_log_posterior(x_obs_B)
p0_B = true_B + 1e-3 * np.random.normal(size=(nwalkers, ndim))
sampler_B = emcee.EnsembleSampler(nwalkers, ndim, log_post_B)
sampler_B.run_mcmc(p0_B, nsteps, progress=True)
mcmc_samples_B = sampler_B.get_chain(discard=nburn, thin=30, flat=True)

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
fname = os.path.join(PLOT_DIR, "posterior_sbi_vs_mcmc_A.png")
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
fname = os.path.join(PLOT_DIR, "posterior_sbi_vs_mcmc_B.png")
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
fname = os.path.join(PLOT_DIR, "posterior_predictive.png")
print(f"Saving {fname}")
fig.savefig(fname, dpi=150, bbox_inches="tight")

if SHOW_FIG:
    plt.show()
