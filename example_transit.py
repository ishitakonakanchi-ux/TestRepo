"""Sanity-check a trained noise-aware NPE, then apply it to real data.

Loads the latest compatible model and compares the amortized NPE posterior
(compress -> flow) against reference NUTS MCMC on the full Gaussian likelihood,
first on two in-distribution mock transits (A, B; known truth), then on one or
more real Kepler DR25 targets. Core models are compared with the matching
nuisance-marginalized columns of the full NUTS posterior.

For a full seven-parameter model, ``NPE_IMPORTANCE_REFINE=1`` enables an
adaptive accuracy mode. The NPE supplies proposals, one vectorized likelihood
pass corrects their weights, and an ESS/max-weight guard rejects unreliable
corrections. This remains much cheaper than a new NUTS chain and exposes when
the amortized proposal does not cover the target posterior.

Usage:
    python example_transit.py [weights_file]
"""
import glob
import csv
import hashlib
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
from scipy.stats import ks_2samp, wasserstein_distance
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_uniform, init_to_value
from numpyro.diagnostics import summary as diagnostic_summary

from npe_wrapper import NPEEstimator
from transit_sbi import (simulator, simulator_batch, score_compress, t_obs,
                         PRIOR_LOW, PRIOR_HIGH, PARAM_LABELS, N_OBS,
                         COMPRESSION_MODES, CORE_MODES)

SEED = 42
PLOT_DIR = "plots"
LIBRARY = "data/dr25_dv_library/dr25_dv_sbi_library.npz"
N_KEPLER = int(os.environ.get("N_KEPLER", "8"))
IMPORTANCE_REFINE_VALUE = os.environ.get("NPE_IMPORTANCE_REFINE", "0")
IMPORTANCE_REFINE = IMPORTANCE_REFINE_VALUE == "1"
IMPORTANCE_BATCH = int(os.environ.get(
    "NPE_IMPORTANCE_BATCH", "10000"
))
IMPORTANCE_MAX_PROPOSALS = int(os.environ.get(
    "NPE_IMPORTANCE_MAX_PROPOSALS", "50000"
))
IMPORTANCE_MIN_PROPOSALS = int(os.environ.get(
    "NPE_IMPORTANCE_MIN_PROPOSALS", "50000"
))
IMPORTANCE_MIN_ESS = float(os.environ.get(
    "NPE_IMPORTANCE_MIN_ESS", "300"
))
IMPORTANCE_MAX_WEIGHT = float(os.environ.get(
    "NPE_IMPORTANCE_MAX_WEIGHT", "0.05"
))
IMPORTANCE_MAX_PARETO_K = float(os.environ.get(
    "NPE_IMPORTANCE_MAX_PARETO_K", "0.7"
))
if N_KEPLER <= 0:
    raise SystemExit("N_KEPLER must be a positive integer")
if IMPORTANCE_REFINE_VALUE not in ("0", "1"):
    raise SystemExit("NPE_IMPORTANCE_REFINE must be 0 or 1")
if IMPORTANCE_BATCH <= 0:
    raise SystemExit("NPE_IMPORTANCE_BATCH must be positive")
if IMPORTANCE_MAX_PROPOSALS < IMPORTANCE_BATCH:
    raise SystemExit(
        "NPE_IMPORTANCE_MAX_PROPOSALS must be at least NPE_IMPORTANCE_BATCH"
    )
if not 0 < IMPORTANCE_MIN_PROPOSALS <= IMPORTANCE_MAX_PROPOSALS:
    raise SystemExit(
        "NPE_IMPORTANCE_MIN_PROPOSALS must be positive and no larger than "
        "NPE_IMPORTANCE_MAX_PROPOSALS"
    )
if (not np.isfinite(IMPORTANCE_MIN_ESS) or IMPORTANCE_MIN_ESS <= 0
        or IMPORTANCE_MIN_ESS > IMPORTANCE_MAX_PROPOSALS):
    raise SystemExit(
        "NPE_IMPORTANCE_MIN_ESS must be positive and no larger than the "
        "proposal budget"
    )
if (not np.isfinite(IMPORTANCE_MAX_WEIGHT)
        or not 0 < IMPORTANCE_MAX_WEIGHT <= 1):
    raise SystemExit("NPE_IMPORTANCE_MAX_WEIGHT must be in (0, 1]")
if not np.isfinite(IMPORTANCE_MAX_PARETO_K):
    raise SystemExit("NPE_IMPORTANCE_MAX_PARETO_K must be finite")
MAX_MCMC_RHAT = 1.01
MIN_MCMC_ESS = 400
MCMC_CACHE_VERSION = 1
LOW, HIGH = np.array(PRIOR_LOW), np.array(PRIOR_HIGH)
LEVELS = 1 - np.exp(-0.5 * np.array([1, 2]) ** 2)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(PLOT_DIR, exist_ok=True)
RESULTS = []

# Two in-distribution mock observations (known truth).
# Params: [b, duration, rp_rs, q1, q2, t0, log10_jitter].
TRUE = {
    "A": np.array([0.35, 0.15, 0.10, 0.30, 0.40, 0.0, np.log10(2e-3)]),
    "B": np.array([0.55, 0.25, 0.17, 0.50, 0.30, 0.0, np.log10(2e-3)]),
}


def _mcmc_fingerprint(x_obs, flux_err, t_grid, init_theta):
    """Hash everything that changes the fixed NUTS reference posterior."""
    digest = hashlib.sha256()
    digest.update(str(MCMC_CACHE_VERSION).encode())
    for value in (x_obs, flux_err, init_theta, LOW, HIGH,
                  np.asarray(t_obs) if t_grid is None else t_grid):
        digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()


def nuts_diagnostics_pass(max_rhat, min_ess, divergences):
    """Return true only for finite diagnostics that pass every NUTS guard."""
    return (
        np.isfinite(max_rhat)
        and np.isfinite(min_ess)
        and max_rhat <= MAX_MCMC_RHAT
        and min_ess >= MIN_MCMC_ESS
        and divergences == 0
    )


def run_mcmc(x_obs, flux_err, init_theta, t_grid=None, cache_key=None):
    fingerprint = _mcmc_fingerprint(x_obs, flux_err, t_grid, init_theta)
    cache_path = None
    if cache_key is not None:
        cache_path = f"data/mcmc_cache/{cache_key}.npz"
        if os.path.exists(cache_path):
            with np.load(cache_path) as cached:
                if cached["fingerprint"].item() == fingerprint:
                    print(f"  Loading cached NUTS reference {cache_path}")
                    return (
                        np.asarray(cached["samples"]),
                        float(cached["max_rhat"]),
                        float(cached["min_ess"]),
                        int(cached["divergences"]),
                        (float(cached["runtime_seconds"])
                         if "runtime_seconds" in cached.files else np.nan),
                    )

    def model(x, err):
        theta = [numpyro.sample(l, dist.Uniform(lo, hi))
                 for l, lo, hi in zip(PARAM_LABELS, PRIOR_LOW, PRIOR_HIGH)]
        flux = (simulator(jnp.stack(theta)) if t_grid is None
                else simulator(jnp.stack(theta), jnp.asarray(t_grid)))
        sigma = jnp.sqrt(err**2 + 10.0 ** (2.0 * theta[6]))
        numpyro.sample("obs", dist.Normal(flux, sigma).to_event(1), obs=x)
    init = np.clip(init_theta, LOW + 1e-6, HIGH - 1e-6)
    def sample(kernel, warmup, draws, chains, seed):
        result = MCMC(
            kernel, num_warmup=warmup, num_samples=draws,
            num_chains=chains, chain_method="vectorized", progress_bar=False,
        )
        result.run(
            jax.random.PRNGKey(seed), x=jnp.asarray(x_obs),
            err=jnp.asarray(flux_err),
        )
        chain_samples = result.get_samples(group_by_chain=True)
        diagnostics = diagnostic_summary(chain_samples, group_by_chain=True)
        max_rhat = max(
            float(diagnostics[lab]["r_hat"]) for lab in PARAM_LABELS
        )
        min_ess = min(
            float(diagnostics[lab]["n_eff"]) for lab in PARAM_LABELS
        )
        divergences = int(np.asarray(
            result.get_extra_fields()["diverging"]
        ).sum())
        return result, max_rhat, min_ess, divergences

    start = perf_counter()
    kernel = NUTS(
        model, dense_mass=True, target_accept_prob=0.9,
        init_strategy=init_to_value(
            values=dict(zip(PARAM_LABELS, map(float, init)))),
    )
    mcmc, max_rhat, min_ess, divergences = sample(
        kernel, 1000, 1000, 2, SEED
    )
    if not nuts_diagnostics_pass(max_rhat, min_ess, divergences):
        print(
            f"  Retrying MCMC: initial max R-hat={max_rhat:.4f}, "
            f"min ESS={min_ess:.0f}, divergences={divergences}"
        )
        robust_kernel = NUTS(
            model, dense_mass=True, target_accept_prob=0.95,
            max_tree_depth=12, init_strategy=init_to_uniform(radius=2.0),
        )
        mcmc, max_rhat, min_ess, divergences = sample(
            robust_kernel, 2000, 2000, 4, SEED + 1
        )
        if not nuts_diagnostics_pass(max_rhat, min_ess, divergences):
            print(
                f"  WARNING: robust MCMC still failed diagnostics: "
                f"max R-hat={max_rhat:.4f}, min ESS={min_ess:.0f}, "
                f"divergences={divergences}"
            )
    s = mcmc.get_samples()
    samples = np.column_stack([np.asarray(s[l]) for l in PARAM_LABELS])
    runtime_seconds = perf_counter() - start
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            samples=samples,
            max_rhat=max_rhat,
            min_ess=min_ess,
            divergences=divergences,
            runtime_seconds=runtime_seconds,
            fingerprint=fingerprint,
        )
    return samples, max_rhat, min_ess, divergences, runtime_seconds


def exact_log_likelihood(samples, x_obs, flux_err, t_grid=None):
    """Evaluate the same Gaussian light-curve likelihood used by NUTS."""
    theta = jnp.asarray(samples)
    if (t_grid is None
            or np.allclose(np.asarray(t_grid), np.asarray(t_obs), atol=1e-7)):
        model_flux = simulator_batch(theta)
    else:
        grid = jnp.asarray(t_grid)
        model_flux = jax.jit(jax.vmap(
            lambda th: simulator(th, grid)
        ))(theta)
    variance = (jnp.asarray(flux_err)[None, :] ** 2
                + 10.0 ** (2.0 * theta[:, -1, None]))
    residual = jnp.asarray(x_obs)[None, :] - model_flux
    return np.asarray(-0.5 * jnp.sum(
        residual**2 / variance + jnp.log(2.0 * jnp.pi * variance), axis=1
    ))


def importance_refine(summary, x_obs, flux_err, t_grid=None,
                      n_output=10_000, rng=None):
    """Correct a full NPE proposal with the exact Gaussian likelihood.

    The prior is uniform inside the same box used to train the NPE, so the
    self-normalized weight is simply ``likelihood / NPE density``. Batches are
    accumulated until the effective sample size is adequate or the proposal
    budget is exhausted. A low ESS or a dominant weight is treated as an
    explicit failure: in that case the raw NPE samples are returned instead of
    pretending that a degenerate correction is trustworthy.
    """
    # ArviZ is comparatively expensive to import; keep it off the pure-NPE
    # startup path, where sub-second warm inference is the main attraction.
    import arviz as az

    start = perf_counter()
    rng = np.random.default_rng(SEED) if rng is None else rng
    raw_proposals, proposals, log_weights = [], [], []
    n_drawn = 0
    ess = 0.0
    max_weight = 1.0
    pareto_k = np.inf
    weights = np.empty(0)
    while n_drawn < IMPORTANCE_MAX_PROPOSALS:
        n_batch = min(
            IMPORTANCE_BATCH, IMPORTANCE_MAX_PROPOSALS - n_drawn
        )
        proposal = npe.sample(
            summary, n_samples=n_batch, show_progress_bars=False
        )
        raw_proposals.append(proposal)
        n_drawn += n_batch
        log_q = npe.log_prob(proposal, summary)
        log_like = exact_log_likelihood(
            proposal, x_obs, flux_err, t_grid
        )
        finite = np.isfinite(log_q) & np.isfinite(log_like)
        if not np.any(finite):
            continue
        proposals.append(proposal[finite])
        log_weights.append(log_like[finite] - log_q[finite])

        joined = np.concatenate(log_weights)
        smoothed, pareto_k = az.psislw(joined)
        weights = np.exp(smoothed)
        total_weight = weights.sum()
        if not np.isfinite(total_weight) or total_weight <= 0:
            ess = 0.0
            max_weight = 1.0
            pareto_k = np.inf
            continue
        weights /= total_weight
        ess = float(1.0 / np.sum(weights**2))
        max_weight = float(weights.max())
        pareto_k = float(np.max(np.asarray(pareto_k)))
        if (n_drawn >= IMPORTANCE_MIN_PROPOSALS
                and ess >= IMPORTANCE_MIN_ESS
                and max_weight <= IMPORTANCE_MAX_WEIGHT
                and pareto_k <= IMPORTANCE_MAX_PARETO_K):
            break

    raw_proposal = np.concatenate(raw_proposals)
    proposal = (np.concatenate(proposals) if proposals
                else np.empty((0, raw_proposal.shape[1])))
    reliable = (
        len(proposal) >= n_output
        and len(proposal) >= IMPORTANCE_MIN_PROPOSALS
        and n_drawn >= IMPORTANCE_MIN_PROPOSALS
        and ess >= IMPORTANCE_MIN_ESS
        and max_weight <= IMPORTANCE_MAX_WEIGHT
        and pareto_k <= IMPORTANCE_MAX_PARETO_K
    )
    if reliable:
        # Systematic resampling has less Monte-Carlo noise than multinomial
        # resampling while preserving the weighted empirical distribution.
        positions = (rng.random() + np.arange(n_output)) / n_output
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, positions, side="right")
        samples = proposal[indices]
    else:
        take = rng.choice(
            len(raw_proposal), size=n_output,
            replace=len(raw_proposal) < n_output,
        )
        samples = raw_proposal[take]
    return samples, {
        "proposals": n_drawn,
        "finite_proposals": len(proposal),
        "ess": ess,
        "max_weight": max_weight,
        "pareto_k": float(pareto_k),
        "reliable": reliable,
        "seconds": perf_counter() - start,
    }


def compare(name, slug, x, flux_err, init, truth=None, t_grid=None):
    """NPE vs MCMC for one observation: print tensions, save a corner plot."""
    start = perf_counter()
    summary = np.array(score_compress(x, flux_err, mode=COMPRESSION))
    compression_seconds = perf_counter() - start
    importance = None
    if IMPORTANCE_REFINE:
        # Keep proposal resampling independent of the global simulator RNG so
        # enabling refinement cannot silently change the later mock curves.
        refine_seed = int.from_bytes(
            hashlib.sha256(slug.encode()).digest()[:8], "little"
        ) ^ SEED
        npe_s, importance = importance_refine(
            summary, x, flux_err, t_grid, n_output=10_000,
            rng=np.random.default_rng(refine_seed),
        )
        npe_seconds = importance["seconds"]
    else:
        start = perf_counter()
        npe_s = npe.sample(summary, n_samples=10_000,
                           show_progress_bars=False)
        npe_seconds = perf_counter() - start
    mcmc_full, max_rhat, min_ess, divergences, mcmc_seconds = run_mcmc(
        x, flux_err, init, t_grid, cache_key=slug
    )
    mcmc_s = mcmc_full[:, TARGET_INDICES]
    nuts_converged = nuts_diagnostics_pass(max_rhat, min_ess, divergences)
    if len(TARGET_INDICES) == len(PARAM_LABELS):
        npe_loglike = exact_log_likelihood(npe_s, x, flux_err, t_grid)
        mcmc_loglike = exact_log_likelihood(mcmc_full, x, flux_err, t_grid)
        likelihood_ks = (
            ks_2samp(npe_loglike, mcmc_loglike).statistic
            if nuts_converged else np.nan
        )
        median_loglike_gap = (
            float(np.median(npe_loglike) - np.median(mcmc_loglike))
            if nuts_converged else np.nan
        )
    else:
        # A core NPE represents the nuisance-marginalized posterior. Its four
        # samples do not define a seven-parameter likelihood point.
        likelihood_ks = median_loglike_gap = np.nan

    method = ("NPE-IS" if importance and importance["reliable"] else "NPE")
    runtime_method = (
        "NPE-IS attempt" if importance and not importance["reliable"]
        else method
    )
    print(
        f"\n=== {name}: {'truth vs ' if truth is not None else ''}"
        f"{method} vs MCMC ==="
    )
    total_npe_seconds = compression_seconds + npe_seconds
    print(
        f"  runtime: compression {compression_seconds:.3f}s + "
        f"{runtime_method} {npe_seconds:.3f}s "
        f"= {total_npe_seconds:.3f}s; MCMC {mcmc_seconds:.3f}s "
        f"({mcmc_seconds / total_npe_seconds:.1f}x speed-up)"
    )
    if importance is not None:
        status = (
            "accepted" if importance["reliable"] else "REJECTED -> raw NPE"
        )
        print(
            f"  importance refinement: {status}; "
            f"ESS {importance['ess']:.0f}/{importance['proposals']}, "
            f"max weight {importance['max_weight']:.4f}, "
            f"Pareto k {importance['pareto_k']:.3f}"
        )
    print(
        f"  MCMC diagnostics: max R-hat {max_rhat:.4f}, "
        f"min ESS {min_ess:.0f}, divergences {divergences}"
    )
    if len(TARGET_INDICES) == len(PARAM_LABELS):
        print(
            f"  joint likelihood: median NPE-MCMC {median_loglike_gap:+.3f}, "
            f"KS {likelihood_ks:.3f}"
        )
    else:
        print("  joint likelihood: not defined for nuisance-marginalized samples")
    if not nuts_converged:
        print("  EXCLUDED: NUTS failed convergence diagnostics")
    truth_target = None if truth is None else np.asarray(truth)[TARGET_INDICES]
    for i, lab in enumerate(TARGET_LABELS):
        npe_mean, npe_std = npe_s[:, i].mean(), npe_s[:, i].std()
        mcmc_mean, mcmc_std = mcmc_s[:, i].mean(), mcmc_s[:, i].std()
        if nuts_converged:
            tn = abs(npe_mean - mcmc_mean) / np.hypot(npe_std, mcmc_std)
            ks = ks_2samp(npe_s[:, i], mcmc_s[:, i])
            ks_statistic, ks_pvalue = ks.statistic, ks.pvalue
            scaled_wasserstein = wasserstein_distance(
                npe_s[:, i], mcmc_s[:, i]
            ) / max(mcmc_std, 1e-12)
        else:
            tn = ks_statistic = ks_pvalue = scaled_wasserstein = np.nan
        tstr = (f"true {truth_target[i]:+.4f}   "
                if truth_target is not None else "")
        print(f"  {lab:9s} {tstr}NPE {npe_mean:+.4f}+/-{npe_std:.4f}"
              f"   MCMC {mcmc_mean:+.4f}+/-{mcmc_std:.4f}"
              f"   tension {tn:.2f}s   KS {ks_statistic:.3f}"
              f"   W/std {scaled_wasserstein:.3f}")
        RESULTS.append({
            "name": name,
            "slug": slug,
            "parameter": lab,
            "truth": "" if truth_target is None else float(truth_target[i]),
            "npe_mean": float(npe_mean),
            "npe_std": float(npe_std),
            "mcmc_mean": float(mcmc_mean),
            "mcmc_std": float(mcmc_std),
            "tension_sigma": float(tn),
            "ks_statistic": float(ks_statistic),
            "ks_pvalue": float(ks_pvalue),
            "wasserstein_over_mcmc_std": float(scaled_wasserstein),
            "compression_seconds": compression_seconds,
            "npe_seconds": npe_seconds,
            "mcmc_seconds": mcmc_seconds,
            "mcmc_max_rhat": max_rhat,
            "mcmc_min_ess": min_ess,
            "mcmc_divergences": divergences,
            "nuts_converged": nuts_converged,
            "median_loglike_gap": median_loglike_gap,
            "likelihood_ks": float(likelihood_ks),
            "posterior_method": (
                method
            ),
            "importance_proposals": (
                importance["proposals"] if importance else 0
            ),
            "importance_ess": importance["ess"] if importance else np.nan,
            "importance_max_weight": (
                importance["max_weight"] if importance else np.nan
            ),
            "importance_pareto_k": (
                importance["pareto_k"] if importance else np.nan
            ),
            "importance_reliable": (
                importance["reliable"] if importance else False
            ),
        })

    # Shared axis range from both sample sets, trimming the 0.5% tails so a few
    # heavy-tail NPE samples don't stretch every panel (keeps NPE/MCMC aligned).
    both = np.concatenate([npe_s, mcmc_s])
    rng = [np.percentile(both[:, i], [0.5, 99.5]) for i in range(both.shape[1])]

    fig = corner.corner(npe_s, labels=TARGET_LABELS, truths=truth_target,
                        color="C0",
                        truth_color="red", smooth=1.0, levels=LEVELS, range=rng,
                        hist_kwargs={"density": True})
    corner.corner(mcmc_s, fig=fig, color="C2", smooth=1.0, levels=LEVELS,
                  range=rng, hist_kwargs={"density": True},
                  contour_kwargs={
                      "linestyles": "-" if nuts_converged else "--"
                  })
    fig.legend([plt.Line2D([], [], color=c) for c in ("C0", "C2")],
               [f"{method} ({COMPRESSION})",
                "NUTS" if nuts_converged else "NUTS (unconverged; excluded)"],
               loc="upper right", fontsize=12)
    fig.suptitle(name, fontsize=11)
    fname = f"{PLOT_DIR}/{OUTPUT_TAG}_{slug}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")


# ── Load the trained NPE ─────────────────────────────────────────────────
if len(sys.argv) > 2:
    sys.exit("Usage: python example_transit.py [weights_file]")
if len(sys.argv) == 2:
    model_fname = sys.argv[1]
else:
    requested = os.environ.get("COMPRESSION")
    if requested is not None and requested not in COMPRESSION_MODES:
        sys.exit(f"Unknown COMPRESSION mode: {requested}")
    modes = (requested,) if requested in COMPRESSION_MODES else COMPRESSION_MODES
    cands = [path for mode in modes
             for path in glob.glob(f"weights/npe_{mode}_20*.pkl")]
    if not cands:
        sys.exit("No noise-aware weights found. Run train_sbi.py first.")
    model_fname = max(cands, key=os.path.getmtime)
print(f"Loading {model_fname}")
npe = NPEEstimator().load(model_fname)
COMPRESSION = npe.metadata_.get("compression")
EXPECTED_SCHEMA = 5 if COMPRESSION in CORE_MODES else 4
if (COMPRESSION not in COMPRESSION_MODES
        or npe.metadata_.get("schema_version") != EXPECTED_SCHEMA):
    sys.exit("This model uses older preprocessing; retrain it.")
TARGET_LABELS = list(npe.metadata_.get("target_labels", PARAM_LABELS))
TARGET_INDICES = list(npe.metadata_.get(
    "target_indices", range(len(PARAM_LABELS))
))
if (len(TARGET_LABELS) != len(TARGET_INDICES)
        or len(set(TARGET_INDICES)) != len(TARGET_INDICES)
        or any(i < 0 or i >= len(PARAM_LABELS) for i in TARGET_INDICES)
        or TARGET_LABELS != [PARAM_LABELS[i] for i in TARGET_INDICES]):
    sys.exit("The model has invalid target-parameter metadata; retrain it.")
if IMPORTANCE_REFINE and len(TARGET_INDICES) != len(PARAM_LABELS):
    sys.exit(
        "NPE importance refinement requires a full seven-parameter model; "
        "use robust weights rather than a core model."
    )
OUTPUT_TAG = COMPRESSION + ("_is" if IMPORTANCE_REFINE else "")

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
if len(holdout) == 0:
    sys.exit("The model has no held-out error profiles; retrain it.")
if (len(np.unique(holdout)) != len(holdout)
        or np.any((holdout < 0) | (holdout >= len(lib["flux_err"])))):
    sys.exit("The model has invalid held-out profile indices; retrain it.")
ranked = holdout[np.argsort(lib["catalog_model_snr"][holdout])]
n_kepler = min(N_KEPLER, len(ranked))
chosen = (ranked[-1:] if n_kepler == 1 else
          ranked[np.linspace(0, len(ranked) - 1, n_kepler, dtype=int)])
for j, i in enumerate(chosen, 1):
    name = str(lib["name"][i])
    x_kep = np.asarray(lib["flux"][i])
    err_kep = np.asarray(lib["flux_err"][i])
    t_kep = np.asarray(lib["phase_time"][i])
    if not np.allclose(t_kep, np.asarray(t_obs), atol=1e-6):
        sys.exit(f"target {i} grid mismatch; rebuild library on the fixed grid")
    init_kep = np.array([0.6, float(lib["dv_duration_hours"][i]) / 24.0,
                         np.sqrt(float(lib["dv_depth_ppm"][i]) * 1e-6),
                         0.3, 0.2, 0.0, np.log10(5e-4)])
    slug = "kepler" if n_kepler == 1 else f"kepler_{j:02d}_i{i:02d}"
    compare(f"Kepler {name}", slug, x_kep, err_kep,
            init=init_kep, t_grid=t_kep)

result_path = f"{PLOT_DIR}/{OUTPUT_TAG}_comparison.csv"
with open(result_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=RESULTS[0])
    writer.writeheader()
    writer.writerows(RESULTS)
print(f"Saved {result_path}")

valid = [row for row in RESULTS if row["nuts_converged"]]
valid_real = [row for row in valid if row["slug"].startswith("kepler")]
all_real = [row for row in RESULTS if row["slug"].startswith("kepler")]
print(
    f"Converged-NUTS subset: {len(valid) // len(TARGET_LABELS)} / "
    f"{len(RESULTS) // len(TARGET_LABELS)} observations"
)
if valid:
    print(
        f"  mean marginal KS="
        f"{np.mean([row['ks_statistic'] for row in valid]):.4f}, "
        f"mean Wasserstein/MCMC-std="
        f"{np.mean([row['wasserstein_over_mcmc_std'] for row in valid]):.4f}"
    )
else:
    print("  no observations passed the NUTS diagnostics")
print(
    f"Converged real-target subset: "
        f"{len(valid_real) // len(TARGET_LABELS)} / "
        f"{len(all_real) // len(TARGET_LABELS)} targets"
)
if valid_real:
    print(
        f"  mean marginal KS="
        f"{np.mean([row['ks_statistic'] for row in valid_real]):.4f}, "
        f"mean Wasserstein/MCMC-std="
        f"{np.mean([row['wasserstein_over_mcmc_std'] for row in valid_real]):.4f}"
    )
