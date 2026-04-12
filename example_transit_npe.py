"""
Example: train an NPE on transit light curves and inspect the posterior.

  1. Simulate training data and train a MAF-based NPE.
  2. Infer posteriors for two different synthetic observations.
  3. Training and validation loss curves.
  4. Corner plot comparing the two posteriors.
  5. Posterior predictive light curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

from ili.utils import Uniform
from npe_wrapper import NPEEstimator
from transit_sbi import (
    simulate_dataset, simulator, t_obs,
    PRIOR_LOW, PRIOR_HIGH, PARAM_LABELS, SIGMA,
)

# ── 1. Simulate training data and train ──────────────────────────────────
theta, x = simulate_dataset(n_sims=500)
prior = Uniform(low=PRIOR_LOW, high=PRIOR_HIGH, device="cpu")

npe = NPEEstimator(
    model="maf",
    hidden_features=50,
    num_transforms=5,
    learning_rate=5e-4,
    batch_size=64,
    stop_after_epochs=20,
)
npe.fit(theta, x, prior)

# ── 2. Training loss vs epoch ────────────────────────────────────────────
summary = npe.summaries_[0]  # first (only) network in the ensemble
fig, ax = plt.subplots()
ax.plot(summary["training_loss"], label="Training")
ax.plot(summary["validation_loss"], label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fig.savefig("training_loss.png", dpi=150, bbox_inches="tight")

# ── 3. Two synthetic observations at different true parameters ───────────
true_A = np.array([0.3, 0.15, 0.10])  # low impact, short duration
true_B = np.array([0.7, 0.30, 0.15])  # high impact, long duration

x_obs_A = simulator(true_A)  # (200,)
x_obs_B = simulator(true_B)  # (200,)

samples_A = npe.sample(x_obs_A, n_samples=10_000)  # (10000, 3)
samples_B = npe.sample(x_obs_B, n_samples=10_000)  # (10000, 3)

# ── 4. Corner plot ───────────────────────────────────────────────────────
# First call creates the figure, second overlays on the same figure.
fig = corner.corner(samples_A, labels=PARAM_LABELS, truths=true_A,
                    color="C0", hist_kwargs={"density": True})
corner.corner(samples_B, fig=fig, truths=true_B,
              color="C1", hist_kwargs={"density": True})
fig.savefig("posterior_corner.png", dpi=150, bbox_inches="tight")

# ── To compare with MCMC (e.g. emcee), overlay MCMC samples on the same
#    corner plot. Assuming `mcmc_samples` is an (n_samples, 3) array from
#    your MCMC chain (after burn-in and thinning):
#
#    corner.corner(mcmc_samples, fig=fig, color="C2",
#                  hist_kwargs={"density": True})


def log_prior_MCMC(theta):
    b, duration, rp_rs = theta
    if (PRIOR_LOW[0] <= b        <= PRIOR_HIGH[0] and
        PRIOR_LOW[1] <= duration <= PRIOR_HIGH[1] and
        PRIOR_LOW[2] <= rp_rs    <= PRIOR_HIGH[2]):
        return 0.0
    return -np.inf

def log_likelihood_MCMC(theta):
    model = simulator(theta)
    return -0.5 * np.sum(
        ((x_obs_A - model) / SIGMA)**2
    )

def log_posterior_MCMC(theta):
    lp = log_prior_MCMC(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_MCMC(theta)


ndim     = 3
nwalkers = 32
nsteps   = 5000 # Took this from TransitMCMC1 code
nburn    = 1000


p0 = true_A + 1e-3 * np.random.normal(
     size=(nwalkers, ndim))
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_MCMC)
sampler.run_mcmc(p0, nsteps, progress=True)
flat_samples = sampler.get_chain(discard=nburn, flat=True)
print("Done.") # TransitMCMC1 code

# Corner Plot of MCMC - NPE
fig = corner.corner(samples_A, labels=PARAM_LABELS, 
                    truths=true_A, color="C0", hist_kwargs={"density": True})
corner.corner(samples_B, fig=fig, truths=true_B, color="C1", 
              hist_kwargs={"density": True})
corner.corner(flat_samples, fig=fig, color="C2",
                   hist_kwargs={"density": True})

fig.savefig("posterior_corner_vs_mcmc.png", dpi=150, bbox_inches="tight")

# ── 5. Posterior predictive light curves ─────────────────────────────────
t_np = np.array(t_obs)
fig, ax = plt.subplots()

for i in range(50):
    ax.plot(t_np, simulator(samples_A[i]), color="C0", alpha=0.05)
ax.plot(t_np, x_obs_A, "k.", ms=2, label="Observed")

ax.set_xlabel(r"$t - t_0$ [days]")
ax.set_ylabel("Relative flux")
ax.legend()
fig.savefig("posterior_predictive.png", dpi=150, bbox_inches="tight")

# ── Convergence test w.r.t. training set size ───────────────────────────
# To check that 500 simulations is enough, retrain the NPE with increasing
# n_sims and compare the posteriors. The idea: if doubling the training set
# doesn't shift the posterior, you have enough simulations.
#
# 1. Pick a fixed observation to evaluate on (e.g. x_obs_A above).
#
# 2. Loop over a range of training set sizes:
        for n in [100, 200, 500, 1000, 2000]:
            theta_n, x_n = simulate_dataset(n_sims=n)
            npe_n = NPEEstimator(...)
            npe_n.fit(theta_n, x_n, prior)
            samples_n = npe_n.sample(x_obs_A, n_samples=10_000)
#
# 3. Compare the posteriors. Some options:
#    - Overlay corner plots for each n (different colours) and check
#      visually whether contours stop shifting.
#    - Track a scalar summary, e.g. the posterior mean or std of each
#      parameter as a function of n_sims, and plot it. Convergence shows
#      as a plateau.
#    - Compare the best validation loss across runs. Once it stops
#      improving with more data, the network has enough training examples.

plt.show()
