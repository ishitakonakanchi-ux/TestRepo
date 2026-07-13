from functools import lru_cache
from pathlib import Path

import numpy as np
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
_edges = jnp.linspace(-WINDOW_DAYS, WINDOW_DAYS, N_OBS + 1)
t_obs = 0.5 * (_edges[:-1] + _edges[1:])

# ═══════════════════════════════════════════════════════════════════
# Fixed value for the parameter we no longer infer:
#   period: cannot be determined from a single phase-folded transit
# ═══════════════════════════════════════════════════════════════════
DEFAULT_PERIOD = 1.0   # days (any reasonable value works for training)
DEFAULT_ERROR_LIBRARY = (
    Path(__file__).parent / "data/dr25_dv_library/dr25_dv_sbi_library.npz"
)
COMPRESSION_MODES = ("weighted", "hybrid")


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


def simulate_dataset(n_sims, noiseless=False, flux_err_profiles=None):
    """Draw parameters and simulate on sampled heteroscedastic error profiles."""
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
    if not noiseless:
        jitter = 10.0 ** theta[:, -1, None]
        sigma = np.sqrt(flux_err**2 + jitter**2)
        x = x + np.random.normal(0, 1, x.shape) * sigma
    return theta, x, flux_err


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
_LOW6 = jnp.asarray(PRIOR_LOW[:6])
_HIGH6 = jnp.asarray(PRIOR_HIGH[:6])
_JITTER_LOW = 10.0 ** PRIOR_LOW[-1]
_JITTER_HIGH = 10.0 ** PRIOR_HIGH[-1]
_DT = float(t_obs[1] - t_obs[0])
_LM_ITERS = 30


def _sim6(th6):
    """Noiseless model from the 6 shape parameters (jitter does not enter)."""
    return simulator(jnp.concatenate([
        th6, jnp.full(1, PRIOR_LOW[-1], dtype=th6.dtype)
    ]))


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


def _gn_fit(x, flux_err):
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
        M = JTJ + lam * jnp.diag(jnp.diag(JTJ)) + 1e-8 * jnp.eye(7)
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

    def fit_from(start):
        par0 = jnp.concatenate([start, jnp.atleast_1d(baseline0)])
        jitter0 = _estimate_jitter(x - _model7(par0), flux_err)
        (par, jitter, _, score), _ = lax.scan(
            body, (par0, jitter0, jnp.asarray(1e-2), nll(par0, jitter0)),
            None, length=_LM_ITERS,
        )
        return par, jitter, score

    pars, jitters, scores = vmap(fit_from)(jnp.stack([th0, high_b]))
    best = jnp.argmin(scores)
    return pars[best], jitters[best]


def _fisher_logsig(par, sigma):
    """Log shape-parameter std from the baseline-marginalised weighted Fisher."""
    Jw = _jac7(par) / sigma[:, None]
    F = Jw.T @ Jw
    Finv = jnp.linalg.inv(F + 1e-6 * jnp.trace(F) / 7.0 * jnp.eye(7))
    return 0.5 * jnp.log(jnp.clip(jnp.diag(Finv)[:6], 1e-24, 1.0))


def _compress_one(x, flux_err):
    par, jitter = _gn_fit(x, flux_err)
    sigma = jnp.sqrt(flux_err**2 + jitter**2)
    resid = x - _model7(par)
    logsig = _fisher_logsig(par, sigma)
    summary = jnp.concatenate([
        par[:6], jnp.atleast_1d(jnp.log10(jitter)), logsig
    ])
    return summary, jnp.clip(resid / sigma, -8.0, 8.0)


@jit
def _compress_weighted_batch(x, flux_err):
    summary, _ = vmap(_compress_one)(x, flux_err)
    return summary


@jit
def _compress_hybrid_batch(x, flux_err):
    summary, white_resid = vmap(_compress_one)(x, flux_err)
    return jnp.concatenate([summary, white_resid], axis=-1)


def score_compress(x, flux_err, mode="weighted"):
    """Compress flux and per-bin errors using weighted or hybrid summaries."""
    if mode not in COMPRESSION_MODES:
        raise ValueError(f"mode must be one of {COMPRESSION_MODES}, got {mode!r}")
    x = np.asarray(x)
    flux_err = np.asarray(flux_err)
    if not np.all(np.isfinite(x)):
        raise ValueError("flux must be finite")
    if not np.all(np.isfinite(flux_err)) or np.any(flux_err <= 0):
        raise ValueError("flux_err must be positive and finite")
    x = jnp.asarray(x)
    flux_err = jnp.asarray(flux_err)
    single = x.ndim == 1
    x_batch = x[None] if single else x
    err_batch = flux_err[None] if flux_err.ndim == 1 else flux_err
    if x_batch.shape != err_batch.shape or x_batch.shape[-1] != N_OBS:
        raise ValueError(
            f"flux and flux_err must have matching (..., {N_OBS}) shapes"
        )
    compress = (
        _compress_weighted_batch if mode == "weighted"
        else _compress_hybrid_batch
    )
    out = compress(x_batch, err_batch)
    return out[0] if single else out


def simulate_compressed(n_sims, mode="weighted", flux_err_profiles=None):
    """Simulate heteroscedastic curves and return theta and chosen summaries."""
    theta, x, flux_err = simulate_dataset(
        n_sims, noiseless=False, flux_err_profiles=flux_err_profiles
    )
    return theta, np.asarray(score_compress(x, flux_err, mode=mode))
