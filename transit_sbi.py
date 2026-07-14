from functools import lru_cache
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, jacfwd, lax
from jaxoplanet.orbits import TransitOrbit
from jaxoplanet.light_curves import limb_dark_light_curve

# Observation grid: bin-centre times of the phase-folded light curve.
# This MUST match build_dr25_dv_library.bin_curve: N_OBS uniform bins over
# [-WINDOW_DAYS, WINDOW_DAYS], flux tabulated at the bin *centres*. Rebuild the
# library with `--window-mode fixed --window-days 0.2 --n-bins 50` so every
# target shares this exact grid.
N_OBS = 50
WINDOW_DAYS = 0.2
COMPRESSION_BATCH_SIZE = 10_000
_edges = jnp.linspace(
    -WINDOW_DAYS, WINDOW_DAYS, N_OBS + 1, dtype=jnp.float32
)
t_obs = 0.5 * (_edges[:-1] + _edges[1:])

# ═══════════════════════════════════════════════════════════════════
# Fixed value for the parameter we no longer infer:
#   period: cannot be determined from a single phase-folded transit
# ═══════════════════════════════════════════════════════════════════
DEFAULT_PERIOD = 1.0   # days (any reasonable value works for training)
DEFAULT_ERROR_LIBRARY = (
    Path(__file__).parent / "data/dr25_dv_library/dr25_dv_sbi_library.npz"
)
COMPRESSION_MODES = (
    "weighted", "hybrid", "full", "embedded", "robust", "core",
    "core_domain",
)
CORE_MODES = ("core", "core_domain")


# Prior bounds for the inferred parameters
#   b        ~ Uniform(0, 0.99)       impact parameter
#   duration ~ Uniform(0.01, 0.35)    transit duration [days]
#   rp_rs    ~ Uniform(0.003, 0.40)   planet-to-star radius ratio
#   q1       ~ Uniform(1e-3, 1)       Kipping LD  (u1 = 2*sqrt(q1)*q2)
#   q2       ~ Uniform(0, 1)          Kipping LD  (u2 = sqrt(q1)*(1-2*q2))
#   t0       ~ Uniform(-0.05, 0.05)   mid-transit time [days]
#   log10_jitter ~ Uniform(-6, -2)     log10 extra white noise
# q1 is floored at 1e-3 because d(u1)/d(q1) ~ 1/sqrt(q1) diverges at q1=0,
# which blows up the score-compression Jacobian and the MCMC gradients there.
PRIOR_LOW = [0.0, 0.01, 0.003, 1e-3, 0.0, -0.05, -6.0]
PRIOR_HIGH = [0.99, 0.35, 0.40, 1.0, 1.0, 0.05, -2.0]
PARAM_LABELS = ["b", "duration", "rp_rs", "q1", "q2", "t0", "log10_jitter"]
CORE_PARAM_INDICES = (0, 1, 2, 5)
CORE_PARAM_LABELS = [PARAM_LABELS[i] for i in CORE_PARAM_INDICES]
CORE_PRIOR_LOW = [PRIOR_LOW[i] for i in CORE_PARAM_INDICES]
CORE_PRIOR_HIGH = [PRIOR_HIGH[i] for i in CORE_PARAM_INDICES]


@jit
def simulator(params, t_grid=None, period=DEFAULT_PERIOD):
    """Simulate a noiseless transit light curve.

    Parameters
    ----------
    params : array (7,)
        [b, duration, rp_rs, q1, q2, t0, log10_jitter] — the 7 parameters
        we infer. (q1, q2) are the Kipping (2013) limb-darkening parameters.
    t_grid : array (N_OBS,) or None
        Time grid for the simulation. If None, uses t_obs default.
    period : float
        Orbital period in days (fixed, not inferred).
    Returns
    -------
    flux : array (N_OBS,)
        Noiseless relative flux evaluated at the time grid.
    """
    b, duration, rp_rs, q1, q2, t0, _log10_jitter = params
    t = t_obs if t_grid is None else t_grid

    # Kipping (2013): (q1, q2) in [0,1]^2 -> physical quadratic LD (u1, u2).
    sq1 = jnp.sqrt(q1)
    u1 = 2.0 * sq1 * q2
    u2 = sq1 * (1.0 - 2.0 * q2)

    orbit = TransitOrbit(
        period=period,
        duration=duration,
        time_transit=t0,
        impact_param=b,
        radius_ratio=rp_rs,
    )
    return 1.0 + limb_dark_light_curve(orbit, [u1, u2])(t)


simulator_batch = jit(vmap(simulator))


@lru_cache(maxsize=None)
def load_flux_err_profiles(path=DEFAULT_ERROR_LIBRARY):
    """Load positive 50-bin error profiles; observed flux is never loaded."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run build_dr25_dv_library.py before training."
        )
    with np.load(path) as library:
        flux_err = np.asarray(library["flux_err"], dtype=np.float64)
    if flux_err.ndim != 2 or flux_err.shape[1] != N_OBS:
        raise ValueError(f"Expected flux_err shape (n, {N_OBS}), got {flux_err.shape}")
    if not np.all(np.isfinite(flux_err)) or np.any(flux_err <= 0):
        raise ValueError("flux_err profiles must be positive and finite")
    return flux_err


def _domain_randomized_observation(model_flux, theta, flux_err):
    """Add moderate unmodelled systematics to otherwise Gaussian curves.

    The reported ``flux_err`` remains the conditioning input, while the true
    white-noise scale is mildly rescaled. Independent, per-curve switches add a
    smooth AR(1) component, a baseline slope, sparse heavy-tailed outliers, and
    occasional missing bins represented by a very large reported uncertainty.
    Thirty-five per cent of curves are left clean so the augmented model retains
    a strong anchor to the explicit Gaussian likelihood used by NUTS.
    """
    n_sims = len(theta)
    reported_err = np.array(flux_err, dtype=np.float64, copy=True)
    jitter = 10.0 ** theta[:, -1, None]
    median_sigma = np.median(
        np.sqrt(reported_err**2 + jitter**2), axis=1, keepdims=True
    )
    augmented = np.random.uniform(size=(n_sims, 1)) >= 0.35

    # The quoted uncertainties may be moderately under- or over-estimated.
    err_scale = np.exp(np.random.normal(0.0, 0.20, size=(n_sims, 1)))
    err_scale = np.clip(err_scale, 0.6, 1.7)
    err_scale = np.where(augmented, err_scale, 1.0)
    true_sigma = np.sqrt((err_scale * reported_err) ** 2 + jitter**2)
    observed = model_flux + np.random.normal(size=model_flux.shape) * true_sigma

    # Short-range correlated variability, normalized to unit marginal scale.
    corr_active = augmented & (np.random.uniform(size=(n_sims, 1)) < 0.60)
    rho = np.random.uniform(0.25, 0.90, size=(n_sims, 1))
    innovation = np.random.normal(size=model_flux.shape)
    correlated = np.empty_like(innovation)
    correlated[:, 0] = innovation[:, 0]
    step_scale = np.sqrt(1.0 - rho[:, 0] ** 2)
    for i in range(1, model_flux.shape[1]):
        correlated[:, i] = (
            rho[:, 0] * correlated[:, i - 1]
            + step_scale * innovation[:, i]
        )
    corr_amp = np.random.uniform(0.05, 0.65, size=(n_sims, 1)) * median_sigma
    observed += correlated * corr_amp * corr_active

    # A shallow detrending error across the fixed transit window.
    slope_active = augmented & (np.random.uniform(size=(n_sims, 1)) < 0.50)
    edge_offset = np.random.normal(0.0, 0.45, size=(n_sims, 1)) * median_sigma
    observed += (
        edge_offset * slope_active
        * (np.asarray(t_obs, dtype=np.float64)[None, :] / WINDOW_DAYS)
    )

    # Sparse cosmic-ray/detrending failures with heavy-tailed amplitudes.
    outlier_active = augmented & (np.random.uniform(size=(n_sims, 1)) < 0.30)
    outlier_rate = np.random.uniform(0.005, 0.035, size=(n_sims, 1))
    outlier_mask = (
        np.random.uniform(size=model_flux.shape) < outlier_rate
    ) & outlier_active
    observed += (
        outlier_mask * np.random.standard_t(3.0, size=model_flux.shape)
        * np.random.uniform(2.0, 4.5, size=(n_sims, 1)) * true_sigma
    )

    # Mark a few unavailable bins by inflating their quoted uncertainty. The
    # flux value then has negligible weight in the deterministic compressor.
    missing_active = augmented & (np.random.uniform(size=(n_sims, 1)) < 0.20)
    missing_rate = np.random.uniform(0.01, 0.06, size=(n_sims, 1))
    missing_mask = (
        np.random.uniform(size=model_flux.shape) < missing_rate
    ) & missing_active
    missing_err = 20.0 * np.max(reported_err, axis=1, keepdims=True)
    reported_err = np.where(missing_mask, missing_err, reported_err)

    metadata = {
        "augmented": augmented[:, 0],
        "error_scale": err_scale[:, 0],
        "correlated": corr_active[:, 0],
        "slope": slope_active[:, 0],
        "outlier_count": outlier_mask.sum(axis=1),
        "missing_count": missing_mask.sum(axis=1),
    }
    return observed, reported_err, metadata


def simulate_dataset(n_sims, noiseless=False, flux_err_profiles=None,
                     noise_model="white", return_noise_metadata=False):
    """Draw parameters and simulate on sampled heteroscedastic error profiles.

    ``noise_model='white'`` exactly retains the schema-4 simulator. The
    ``'domain'`` option mixes clean curves with moderate unmodelled systematics
    so nuisance-marginalized NPEs can learn robustness to real-data mismatch.
    """
    if (not isinstance(n_sims, (int, np.integer))
            or isinstance(n_sims, (bool, np.bool_)) or n_sims <= 0):
        raise ValueError("n_sims must be a positive integer")
    if noise_model not in ("white", "domain"):
        raise ValueError("noise_model must be 'white' or 'domain'")
    profiles = (
        load_flux_err_profiles()
        if flux_err_profiles is None
        else np.asarray(flux_err_profiles, dtype=np.float64)
    )
    if profiles.ndim != 2 or profiles.shape[1] != N_OBS:
        raise ValueError(f"Expected flux_err_profiles shape (n, {N_OBS})")
    if len(profiles) == 0:
        raise ValueError("flux_err_profiles must contain at least one profile")
    if not np.all(np.isfinite(profiles)) or np.any(profiles <= 0):
        raise ValueError("flux_err_profiles must be positive and finite")

    theta = np.column_stack([
        np.random.uniform(PRIOR_LOW[i], PRIOR_HIGH[i], n_sims)
        for i in range(7)
    ])
    pair = np.random.randint(0, len(profiles), size=(2, n_sims))
    mix = np.random.uniform(0.0, 1.0, size=(n_sims, 1))
    flux_err = np.exp(
        (1.0 - mix) * np.log(profiles[pair[0]])
        + mix * np.log(profiles[pair[1]])
    )
    x = np.asarray(simulator_batch(jnp.array(theta)))
    noise_metadata = None
    if not noiseless:
        if noise_model == "domain":
            x, flux_err, noise_metadata = _domain_randomized_observation(
                x, theta, flux_err
            )
        else:
            jitter = 10.0 ** theta[:, -1, None]
            sigma = np.sqrt(flux_err**2 + jitter**2)
            x = x + np.random.normal(0, 1, x.shape) * sigma
    result = (theta, x, flux_err)
    return (*result, noise_metadata) if return_noise_metadata else result


# ═══════════════════════════════════════════════════════════════════
# Noise-aware score compression: flux + flux_err -> 13 weighted summaries:
#   6  approximate-MLE shape parameters (bounded adaptive Levenberg-Marquardt
#      fit, seeded by a moment-based guess) — the posterior *location*,
#   1  fitted log10 extra jitter, with sigma_i^2 = flux_err_i^2 + jitter^2,
#   6  log marginal std from the weighted Fisher F = J^T W J.
# The hybrid representation appends the 50 whitened residuals. A baseline offset
# is profiled as a nuisance parameter in both the fit and Fisher matrix.
# Re-fitting per target keeps the compression near-sufficient across the whole
# prior (not just near one fiducial). The map is deterministic and applied
# identically to simulations and real data, so the NPE learns p(theta | summary).
# ═══════════════════════════════════════════════════════════════════
SUMMARY_LABELS = ([f"hat_{p}" for p in PARAM_LABELS[:6]] + ["hat_log10_jitter"]
                  + [f"logsig_{p}" for p in PARAM_LABELS[:6]])
HYBRID_LABELS = SUMMARY_LABELS + [f"white_resid_{i}" for i in range(N_OBS)]
FULL_LABELS = (SUMMARY_LABELS + ["hat_baseline"]
               + [f"white_resid_{i}" for i in range(N_OBS)]
               + [f"log10_flux_err_{i}" for i in range(N_OBS)])
_LOW6 = jnp.asarray(PRIOR_LOW[:6], dtype=jnp.float32)
_HIGH6 = jnp.asarray(PRIOR_HIGH[:6], dtype=jnp.float32)
_JITTER_LOW = jnp.asarray(10.0 ** PRIOR_LOW[-1], dtype=jnp.float32)
_JITTER_HIGH = jnp.asarray(10.0 ** PRIOR_HIGH[-1], dtype=jnp.float32)
_DT = float(t_obs[1] - t_obs[0])
_LM_ITERS = 30


def _sim6(th6):
    """Noiseless model from the 6 shape parameters (jitter does not enter)."""
    flux = simulator(jnp.concatenate([
        th6, jnp.full(1, PRIOR_LOW[-1], dtype=th6.dtype)
    ]))
    return flux.astype(th6.dtype)


def _init6(x):
    """Cheap moment-based guess to seed the fit in the right basin."""
    dip = jnp.clip(1.0 - x, 0.0, None)
    depth = jnp.quantile(dip, 0.95)
    rp = jnp.sqrt(jnp.clip(depth, 1e-6, None))
    dur = jnp.sum(dip > 0.5 * depth) * _DT
    t0 = jnp.sum(dip * t_obs) / (jnp.sum(dip) + 1e-12)
    return jnp.clip(jnp.array([0.5, dur, rp, 0.5, 0.5, t0]), _LOW6, _HIGH6)


def _model7(par):
    """Transit model with a fitted additive baseline nuisance parameter."""
    return _sim6(par[:6]) + par[6]


_jac7 = jacfwd(_model7)                   # (N_OBS, 7)


def _estimate_jitter(resid, flux_err):
    excess_variance = jnp.mean(resid**2 - flux_err**2)
    return jnp.sqrt(jnp.clip(
        excess_variance, _JITTER_LOW**2, _JITTER_HIGH**2
    ))


def _gn_fit(x, flux_err, robust=False):
    """Weighted approximate MLE of shape, baseline, and extra jitter."""
    def nll(par, jitter):
        resid = x - _model7(par)
        variance = flux_err**2 + jitter**2
        return 0.5 * jnp.sum(resid**2 / variance + jnp.log(variance))

    def body(carry, _):
        par, jitter, lam, nll0 = carry
        sigma = jnp.sqrt(flux_err**2 + jitter**2)
        Jw = _jac7(par) / sigma[:, None]
        rw = (x - _model7(par)) / sigma
        JTJ = Jw.T @ Jw
        M = (JTJ + lam * jnp.diag(jnp.diag(JTJ))
             + 1e-8 * jnp.eye(7, dtype=par.dtype))
        candidate = par + jnp.linalg.solve(M, Jw.T @ rw)
        par_new = jnp.concatenate([
            jnp.clip(candidate[:6], _LOW6, _HIGH6),
            jnp.clip(candidate[6:], -0.05, 0.05),
        ])
        jitter_new = _estimate_jitter(x - _model7(par_new), flux_err)
        nll_new = nll(par_new, jitter_new)
        ok = nll_new < nll0
        return (jnp.where(ok, par_new, par),
                jnp.where(ok, jitter_new, jitter),
                jnp.clip(jnp.where(ok, lam * 0.5, lam * 3.0), 1e-7, 1e7),
                jnp.where(ok, nll_new, nll0)), None

    outer = (jnp.abs(t_obs) > 0.15).astype(x.dtype)
    baseline0 = jnp.sum(outer * (x - 1.0)) / jnp.sum(outer)
    th0 = _init6(x - baseline0)
    high_b = th0.at[0].set(0.9).at[1].set(
        jnp.clip(2.0 * th0[1], _LOW6[1], _HIGH6[1])
    )
    # Two additional duration/b seeds expose separated grazing/non-grazing
    # likelihood basins. They are opt-in so schema-4 models trained with the
    # original two-start map remain application-compatible.
    moderate_long = th0.at[0].set(0.2).at[1].set(
        jnp.clip(1.15 * th0[1], _LOW6[1], _HIGH6[1])
    )
    central_long = th0.at[0].set(0.5).at[1].set(
        jnp.clip(1.3 * th0[1], _LOW6[1], _HIGH6[1])
    )

    def fit_from(start):
        par0 = jnp.concatenate([start, jnp.atleast_1d(baseline0)])
        jitter0 = _estimate_jitter(x - _model7(par0), flux_err)
        (par, jitter, _, score), _ = lax.scan(
            body, (par0, jitter0, jnp.asarray(1e-2, dtype=par0.dtype),
                   nll(par0, jitter0)),
            None, length=_LM_ITERS,
        )
        return par, jitter, score

    starts = ([th0, high_b, moderate_long, central_long]
              if robust else [th0, high_b])
    pars, jitters, scores = vmap(fit_from)(jnp.stack(starts))
    best = jnp.argmin(scores)
    return pars[best], jitters[best]


def _fisher_logsig(par, sigma):
    """Log shape-parameter std from the baseline-marginalised weighted Fisher."""
    Jw = _jac7(par) / sigma[:, None]
    F = Jw.T @ Jw
    Finv = jnp.linalg.inv(
        F + 1e-6 * jnp.trace(F) / 7.0 * jnp.eye(7, dtype=par.dtype)
    )
    return 0.5 * jnp.log(jnp.clip(jnp.diag(Finv)[:6], 1e-24, 1.0))


def _compress_one(x, flux_err):
    par, jitter = _gn_fit(x, flux_err)
    sigma = jnp.sqrt(flux_err**2 + jitter**2)
    resid = x - _model7(par)
    logsig = _fisher_logsig(par, sigma)
    summary = jnp.concatenate([
        par[:6], jnp.atleast_1d(jnp.log10(jitter)), logsig
    ])
    return summary, jnp.clip(resid / sigma, -8.0, 8.0), par[6]


def _compress_one_robust(x, flux_err):
    """Full representation using a four-start likelihood fit."""
    par, jitter = _gn_fit(x, flux_err, robust=True)
    sigma = jnp.sqrt(flux_err**2 + jitter**2)
    resid = x - _model7(par)
    logsig = _fisher_logsig(par, sigma)
    summary = jnp.concatenate([
        par[:6], jnp.atleast_1d(jnp.log10(jitter)), logsig
    ])
    return summary, jnp.clip(resid / sigma, -8.0, 8.0), par[6]


@jit
def _compress_weighted_batch(x, flux_err):
    summary, _, _ = vmap(_compress_one)(x, flux_err)
    return summary


@jit
def _compress_hybrid_batch(x, flux_err):
    summary, white_resid, _ = vmap(_compress_one)(x, flux_err)
    return jnp.concatenate([summary, white_resid], axis=-1)


@jit
def _compress_full_batch(x, flux_err):
    """Near-lossless conditioning for full-likelihood posterior recovery."""
    summary, white_resid, baseline = vmap(_compress_one)(x, flux_err)
    return jnp.concatenate([
        summary,
        baseline[:, None],
        white_resid,
        jnp.log10(flux_err),
    ], axis=-1)


@jit
def _compress_robust_batch(x, flux_err):
    """Near-lossless context anchored to the best of four fit basins."""
    summary, white_resid, baseline = vmap(_compress_one_robust)(x, flux_err)
    return jnp.concatenate([
        summary,
        baseline[:, None],
        white_resid,
        jnp.log10(flux_err),
    ], axis=-1)


def score_compress(x, flux_err, mode="weighted"):
    """Compress flux and errors with the requested noise-aware representation.

    ``embedded`` uses the same near-lossless inputs as ``full``; its learned
    dimensionality reduction is part of the NPE rather than this deterministic
    preprocessing step. ``robust`` combines that learned embedding with a
    four-start fit that exposes separated transit-geometry basins.
    """
    if mode not in COMPRESSION_MODES:
        raise ValueError(f"mode must be one of {COMPRESSION_MODES}, got {mode!r}")
    x = np.asarray(x)
    flux_err = np.asarray(flux_err)
    if not np.all(np.isfinite(x)):
        raise ValueError("flux must be finite")
    if not np.all(np.isfinite(flux_err)) or np.any(flux_err <= 0):
        raise ValueError("flux_err must be positive and finite")
    # Training uses JAX float32. Keep the compression map in that same dtype
    # even when callers enable float64 for NUTS; otherwise the bounded LM fit
    # can take a different branch and produce out-of-distribution summaries.
    x = jnp.asarray(x, dtype=jnp.float32)
    flux_err = jnp.asarray(flux_err, dtype=jnp.float32)
    single = x.ndim == 1
    x_batch = x[None] if single else x
    err_batch = flux_err[None] if flux_err.ndim == 1 else flux_err
    if x_batch.shape != err_batch.shape or x_batch.shape[-1] != N_OBS:
        raise ValueError(
            f"flux and flux_err must have matching (..., {N_OBS}) shapes"
        )
    compress = {
        "weighted": _compress_weighted_batch,
        "hybrid": _compress_hybrid_batch,
        "full": _compress_full_batch,
        "embedded": _compress_full_batch,
        "robust": _compress_robust_batch,
        "core": _compress_robust_batch,
        "core_domain": _compress_robust_batch,
    }[mode]
    # `example_transit.py` enables x64 for NUTS. Compile and execute the
    # compressor under the training-time x64 setting so its branch decisions
    # are identical in training, validation, and application.
    with jax.enable_x64(False):
        out = compress(x_batch, err_batch)
    return out[0] if single else out


def simulate_compressed(n_sims, mode="weighted", flux_err_profiles=None):
    """Simulate heteroscedastic curves and return theta and chosen summaries."""
    if mode not in COMPRESSION_MODES:
        raise ValueError(f"mode must be one of {COMPRESSION_MODES}, got {mode!r}")
    theta, x, flux_err = simulate_dataset(
        n_sims, noiseless=False, flux_err_profiles=flux_err_profiles,
        noise_model="domain" if mode == "core_domain" else "white",
    )
    target = theta[:, CORE_PARAM_INDICES] if mode in CORE_MODES else theta
    # A single 150k-example robust XLA graph uses roughly 12 GB of host memory.
    # Compression is row-wise, so fixed-size chunks are numerically equivalent
    # while compiling only one much smaller reusable graph.
    summaries = [
        np.asarray(score_compress(
            x[start:start + COMPRESSION_BATCH_SIZE],
            flux_err[start:start + COMPRESSION_BATCH_SIZE], mode=mode,
        ))
        for start in range(0, n_sims, COMPRESSION_BATCH_SIZE)
    ]
    return target, np.concatenate(summaries, axis=0)
