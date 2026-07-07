"""Sanity-check the trained MLE-compression NPE, then apply it to real data.

Loads the latest weights/npe_mle_*.pkl and compares the amortized NPE posterior
(compress -> flow) against reference NUTS MCMC on the full Gaussian likelihood,
first on two in-distribution mock transits (A, B; known truth), then on one real
Kepler DR25 target.

Usage:
    python example_transit.py [weights_file]
"""
import glob
import os
import sys
import logging
import warnings

logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="sbi")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
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
from transit_sbi import (simulator, score_compress, t_obs,
                         PRIOR_LOW, PRIOR_HIGH, PARAM_LABELS, N_OBS)

SEED = 42
PLOT_DIR = "plots"
LIBRARY = "data/dr25_dv_library/dr25_dv_sbi_library.npz"
KEPLER_INDEX = 0
LOW, HIGH = np.array(PRIOR_LOW), np.array(PRIOR_HIGH)
LEVELS = 1 - np.exp(-0.5 * np.array([1, 2]) ** 2)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(PLOT_DIR, exist_ok=True)

# Two in-distribution mock observations (known truth).
# Params: [b, duration, rp_rs, q1, q2, t0, scatter]; q1,q2 are Kipping LD.
TRUE = {
    "A": np.array([0.35, 0.15, 0.10, 0.30, 0.40, 0.0, 2e-3]),  # low impact, short
    "B": np.array([0.55, 0.25, 0.17, 0.50, 0.30, 0.0, 2e-3]),  # higher impact, longer
}


def run_mcmc(x_obs, init_theta, t_grid=None):
    def model(x):
        theta = [numpyro.sample(l, dist.Uniform(lo, hi))
                 for l, lo, hi in zip(PARAM_LABELS, PRIOR_LOW, PRIOR_HIGH)]
        flux = (simulator(jnp.stack(theta)) if t_grid is None
                else simulator(jnp.stack(theta), jnp.asarray(t_grid)))
        numpyro.sample("obs", dist.Normal(flux, theta[6]).to_event(1), obs=x)
    init = np.clip(init_theta, LOW + 1e-6, HIGH - 1e-6)
    kernel = NUTS(model, dense_mass=True, target_accept_prob=0.9,
                  init_strategy=init_to_value(
                      values=dict(zip(PARAM_LABELS, map(float, init)))))
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=2,
                chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(SEED), x=jnp.asarray(x_obs))
    s = mcmc.get_samples()
    return np.column_stack([np.asarray(s[l]) for l in PARAM_LABELS])


def compare(name, slug, x, init, truth=None, t_grid=None):
    """NPE vs MCMC for one observation: print tensions, save a corner plot."""
    npe_s = npe.sample(np.array(score_compress(x)), n_samples=10_000,
                       show_progress_bars=False)
    mcmc_s = run_mcmc(x, init, t_grid)

    print(f"\n=== {name}: {'truth vs ' if truth is not None else ''}NPE vs MCMC ===")
    for i, lab in enumerate(PARAM_LABELS):
        tn = abs(npe_s[:, i].mean() - mcmc_s[:, i].mean()) / np.hypot(
            npe_s[:, i].std(), mcmc_s[:, i].std())
        tstr = f"true {truth[i]:+.4f}   " if truth is not None else ""
        print(f"  {lab:9s} {tstr}NPE {npe_s[:, i].mean():+.4f}+/-{npe_s[:, i].std():.4f}"
              f"   MCMC {mcmc_s[:, i].mean():+.4f}+/-{mcmc_s[:, i].std():.4f}"
              f"   tension {tn:.2f}s")

    # Shared axis range from both sample sets, trimming the 0.5% tails so a few
    # heavy-tail NPE samples don't stretch every panel (keeps NPE/MCMC aligned).
    both = np.concatenate([npe_s, mcmc_s])
    rng = [np.percentile(both[:, i], [0.5, 99.5]) for i in range(both.shape[1])]

    fig = corner.corner(npe_s, labels=PARAM_LABELS, truths=truth, color="C0",
                        truth_color="red", smooth=1.0, levels=LEVELS, range=rng,
                        hist_kwargs={"density": True})
    corner.corner(mcmc_s, fig=fig, color="C2", smooth=1.0, levels=LEVELS,
                  range=rng, hist_kwargs={"density": True})
    fig.legend([plt.Line2D([], [], color=c) for c in ("C0", "C2")],
               ["NPE (MLE compression)", "MCMC"], loc="upper right", fontsize=12)
    fig.suptitle(name, fontsize=11)
    fname = f"{PLOT_DIR}/mle_{slug}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")


# ── Load the trained NPE ─────────────────────────────────────────────────
if len(sys.argv) == 2:
    model_fname = sys.argv[1]
else:
    cands = glob.glob("weights/npe_mle_*.pkl")
    if not cands:
        sys.exit("No weights/npe_mle_*.pkl found. Run train_sbi.py first.")
    model_fname = max(cands, key=os.path.getmtime)
print(f"Loading {model_fname}")
npe = NPEEstimator().load(model_fname)

# ── 1. Two synthetic mocks (known truth) ─────────────────────────────────
for tag, true in TRUE.items():
    x = np.array(simulator(true)) + np.random.normal(0, true[-1], N_OBS)
    compare(f"Mock {tag}", f"mock_{tag}", x, init=true, truth=true)

# ── 2. One real Kepler target ────────────────────────────────────────────
lib = np.load(LIBRARY, allow_pickle=True)
i = KEPLER_INDEX
name = str(lib["name"][i])
x_kep = np.asarray(lib["flux"][i])
t_kep = np.asarray(lib["phase_time"][i])
assert np.allclose(t_kep, np.asarray(t_obs), atol=1e-6), (
    f"target {i} grid mismatch; rebuild library on the fixed grid")
init_kep = np.array([0.6, float(lib["dv_duration_hours"][i]) / 24.0,
                     np.sqrt(float(lib["dv_depth_ppm"][i]) * 1e-6),
                     0.3, 0.2, 0.0, 5e-4])
compare(f"Kepler {name}", "kepler", x_kep, init=init_kep, t_grid=t_kep)
