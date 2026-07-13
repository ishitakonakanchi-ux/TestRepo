"""Sanity-check a trained noise-aware NPE, then apply it to real data.

Loads the latest weighted/hybrid model and compares the amortized NPE posterior
(compress -> flow) against reference NUTS MCMC on the full Gaussian likelihood,
first on two in-distribution mock transits (A, B; known truth), then on one or
more real Kepler DR25 targets.

Usage:
    python example_transit.py [weights_file]
"""
import glob
import os
import sys
import logging
import warnings
from time import perf_counter

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
                         PRIOR_LOW, PRIOR_HIGH, PARAM_LABELS, N_OBS,
                         COMPRESSION_MODES)

SEED = 42
PLOT_DIR = "plots"
LIBRARY = "data/dr25_dv_library/dr25_dv_sbi_library.npz"
N_KEPLER = max(1, int(os.environ.get("N_KEPLER", "1")))
LOW, HIGH = np.array(PRIOR_LOW), np.array(PRIOR_HIGH)
LEVELS = 1 - np.exp(-0.5 * np.array([1, 2]) ** 2)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(PLOT_DIR, exist_ok=True)

# Two in-distribution mock observations (known truth).
# Params: [b, duration, rp_rs, q1, q2, t0, log10_jitter].
TRUE = {
    "A": np.array([0.35, 0.15, 0.10, 0.30, 0.40, 0.0, np.log10(2e-3)]),
    "B": np.array([0.55, 0.25, 0.17, 0.50, 0.30, 0.0, np.log10(2e-3)]),
}


def run_mcmc(x_obs, flux_err, init_theta, t_grid=None):
    def model(x, err):
        theta = [numpyro.sample(l, dist.Uniform(lo, hi))
                 for l, lo, hi in zip(PARAM_LABELS, PRIOR_LOW, PRIOR_HIGH)]
        flux = (simulator(jnp.stack(theta)) if t_grid is None
                else simulator(jnp.stack(theta), jnp.asarray(t_grid)))
        sigma = jnp.sqrt(err**2 + 10.0 ** (2.0 * theta[6]))
        numpyro.sample("obs", dist.Normal(flux, sigma).to_event(1), obs=x)
    init = np.clip(init_theta, LOW + 1e-6, HIGH - 1e-6)
    kernel = NUTS(model, dense_mass=True, target_accept_prob=0.9,
                  init_strategy=init_to_value(
                      values=dict(zip(PARAM_LABELS, map(float, init)))))
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=2,
                chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(SEED), x=jnp.asarray(x_obs),
             err=jnp.asarray(flux_err))
    s = mcmc.get_samples()
    return np.column_stack([np.asarray(s[l]) for l in PARAM_LABELS])


def compare(name, slug, x, flux_err, init, truth=None, t_grid=None):
    """NPE vs MCMC for one observation: print tensions, save a corner plot."""
    start = perf_counter()
    summary = np.array(score_compress(x, flux_err, mode=COMPRESSION))
    compression_seconds = perf_counter() - start
    start = perf_counter()
    npe_s = npe.sample(summary, n_samples=10_000,
                       show_progress_bars=False)
    npe_seconds = perf_counter() - start
    start = perf_counter()
    mcmc_s = run_mcmc(x, flux_err, init, t_grid)
    mcmc_seconds = perf_counter() - start

    print(f"\n=== {name}: {'truth vs ' if truth is not None else ''}NPE vs MCMC ===")
    total_npe_seconds = compression_seconds + npe_seconds
    print(
        f"  runtime: compression {compression_seconds:.3f}s + NPE {npe_seconds:.3f}s "
        f"= {total_npe_seconds:.3f}s; MCMC {mcmc_seconds:.3f}s "
        f"({mcmc_seconds / total_npe_seconds:.1f}x speed-up)"
    )
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
               [f"NPE ({COMPRESSION})", "MCMC"], loc="upper right", fontsize=12)
    fig.suptitle(name, fontsize=11)
    fname = f"{PLOT_DIR}/{COMPRESSION}_{slug}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")


# ── Load the trained NPE ─────────────────────────────────────────────────
if len(sys.argv) == 2:
    model_fname = sys.argv[1]
else:
    cands = (glob.glob("weights/npe_weighted_*.pkl")
             + glob.glob("weights/npe_hybrid_*.pkl"))
    if not cands:
        sys.exit("No noise-aware weights found. Run train_sbi.py first.")
    model_fname = max(cands, key=os.path.getmtime)
print(f"Loading {model_fname}")
npe = NPEEstimator().load(model_fname)
COMPRESSION = npe.metadata_.get("compression")
if (COMPRESSION not in COMPRESSION_MODES
        or npe.metadata_.get("schema_version") != 4):
    sys.exit("This model uses older preprocessing; retrain it.")

# ── 1. Two synthetic mocks (known truth) ─────────────────────────────────
for tag, true in TRUE.items():
    flux_err = np.full(N_OBS, 5e-4)
    sigma = np.sqrt(flux_err**2 + 10.0 ** (2.0 * true[-1]))
    x = np.array(simulator(true)) + np.random.normal(0, sigma, N_OBS)
    compare(f"Mock {tag}", f"mock_{tag}", x, flux_err,
            init=true, truth=true)

# ── 2. Held-out real Kepler targets spanning catalog S/N ─────────────────
lib = np.load(LIBRARY, allow_pickle=True)
expected_profiles = npe.metadata_.get("error_profile_count")
if expected_profiles is not None and len(lib["flux_err"]) != expected_profiles:
    sys.exit("The Kepler library changed after training; retrain the model.")
holdout = np.asarray(
    npe.metadata_.get("holdout_profile_indices", []), dtype=int
)
ranked = (holdout[np.argsort(lib["catalog_model_snr"][holdout])]
          if len(holdout) else np.array([0]))
n_kepler = min(N_KEPLER, len(ranked))
chosen = (ranked[-1:] if n_kepler == 1 else
          ranked[np.linspace(0, len(ranked) - 1, n_kepler, dtype=int)])
for j, i in enumerate(chosen, 1):
    name = str(lib["name"][i])
    x_kep = np.asarray(lib["flux"][i])
    err_kep = np.asarray(lib["flux_err"][i])
    t_kep = np.asarray(lib["phase_time"][i])
    assert np.allclose(t_kep, np.asarray(t_obs), atol=1e-6), (
        f"target {i} grid mismatch; rebuild library on the fixed grid")
    init_kep = np.array([0.6, float(lib["dv_duration_hours"][i]) / 24.0,
                         np.sqrt(float(lib["dv_depth_ppm"][i]) * 1e-6),
                         0.3, 0.2, 0.0, np.log10(5e-4)])
    slug = "kepler" if n_kepler == 1 else f"kepler_{j:02d}_i{i:02d}"
    compare(f"Kepler {name}", slug, x_kep, err_kep,
            init=init_kep, t_grid=t_kep)
