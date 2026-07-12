import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, jacfwd, lax
from jaxoplanet.orbits import TransitOrbit
from jaxoplanet.light_curves import limb_dark_light_curve

SIGMA = 5e-4  # Gaussian noise std on flux

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
# Fixed values for parameters we no longer infer:
#   period: cannot be determined from a single phase-folded transit
#   t0:     phase-folding already centers the transit at t=0
# ═══════════════════════════════════════════════════════════════════
DEFAULT_PERIOD = 1.0   # days (any reasonable value works for training)


# Prior bounds for the inferred parameters
#   b        ~ Uniform(0, 0.99)       impact parameter
#   duration ~ Uniform(0.01, 0.35)    transit duration [days]
#   rp_rs    ~ Uniform(0.03, 0.40)    planet-to-star radius ratio
#   q1       ~ Uniform(1e-3, 1)       Kipping LD  (u1 = 2*sqrt(q1)*q2)
#   q2       ~ Uniform(0, 1)          Kipping LD  (u2 = sqrt(q1)*(1-2*q2))
#   t0       ~ Uniform(-0.05, 0.05)   mid-transit time [days]
#   scatter  ~ Uniform(1e-4, 1e-2)    observational noise level
# q1 is floored at 1e-3 because d(u1)/d(q1) ~ 1/sqrt(q1) diverges at q1=0,
# which blows up the score-compression Jacobian and the MCMC gradients there.
PRIOR_LOW = [0.0, 0.01, 0.03, 1e-3, 0.0, -0.05, 1e-4]
PRIOR_HIGH = [0.99, 0.35, 0.40, 1.0, 1.0, 0.05, 1e-2]
PARAM_LABELS = ["b", "duration", "rp_rs", "q1", "q2", "t0", "scatter"]



@jit
def simulator(params, t_grid=None, period=DEFAULT_PERIOD):
    """Simulate a noiseless transit light curve.

    Parameters
    ----------
    params : array (7,)
        [b, duration, rp_rs, q1, q2, t0, scatter] — the 7 transit parameters
        we infer. (q1, q2) are the Kipping (2013) limb-darkening parameters.
    t_grid : array (N_OBS,) or None
        Time grid for the simulation. If None, uses t_obs default.
    period : float
        Orbital period in days (fixed, not inferred).
    t0 : float
        Mid-transit time in days (fixed, not inferred).

    Returns
    -------
    flux : array (N_OBS,)
        Noiseless relative flux evaluated at the time grid.
    """
    b, duration, rp_rs, q1, q2, t0, scatter = params
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



def simulate_dataset(n_sims, noiseless=False):
    """Draw parameters from the prior and simulate light curves."""
    theta = np.column_stack([
        np.random.uniform(PRIOR_LOW[i], PRIOR_HIGH[i], n_sims)
        for i in range(7)   
    ])
    x = np.asarray(simulator_batch(jnp.array(theta)))
    if not noiseless: #per simulation scatter used here
        scatter_vals = theta[:, -1]
        noise = np.random.normal(0, 1, x.shape) * scatter_vals[:, None]
        x = x + noise
    return theta, x


# ═══════════════════════════════════════════════════════════════════
# Nonlinear score compression: raw flux (N_OBS) -> 13 near-sufficient summaries:
#   6  approximate-MLE shape parameters (bounded adaptive Levenberg-Marquardt
#      fit, seeded by a moment-based guess) — the posterior *location*,
#   1  residual RMS to the fit — the noise level, defined exactly as MCMC's
#      likelihood scatter,
#   6  log per-parameter marginal std from the fit's Fisher F = J^T J / sigma^2 —
#      the posterior *widths*.
# Re-fitting per target keeps the compression near-sufficient across the whole
# prior (not just near one fiducial). The map is deterministic and applied
# identically to simulations and real data, so the NPE learns p(theta | summary).
# ═══════════════════════════════════════════════════════════════════
SUMMARY_LABELS = ([f"hat_{p}" for p in PARAM_LABELS[:6]] + ["resid_rms"]
                  + [f"logsig_{p}" for p in PARAM_LABELS[:6]])
_LOW6 = jnp.asarray(PRIOR_LOW[:6])
_HIGH6 = jnp.asarray(PRIOR_HIGH[:6])
_DT = float(t_obs[1] - t_obs[0])
_LM_ITERS = 30


def _sim6(th6):
    """Noiseless model from the 6 shape parameters (scatter does not enter)."""
    return simulator(jnp.concatenate([th6, jnp.zeros(1, th6.dtype)]))


_jac6 = jacfwd(_sim6)                   # (N_OBS, 6)


def _init6(x):
    """Cheap moment-based guess to seed the fit in the right basin."""
    dip = jnp.clip(1.0 - x, 0.0, None)
    depth = jnp.quantile(dip, 0.95)
    rp = jnp.sqrt(jnp.clip(depth, 1e-6, None))
    dur = jnp.sum(dip > 0.5 * depth) * _DT
    t0 = jnp.sum(dip * t_obs) / (jnp.sum(dip) + 1e-12)
    return jnp.clip(jnp.array([0.5, dur, rp, 0.5, 0.5, t0]), _LOW6, _HIGH6)


def _gn_fit(x):
    """Approximate MLE of the 6 shape params via bounded, adaptive LM."""
    def ssr(th):
        r = x - _sim6(th)
        return jnp.sum(r * r)

    def body(carry, _):
        th, lam, s0 = carry
        J = _jac6(th)
        JTJ = J.T @ J
        M = JTJ + lam * jnp.diag(jnp.diag(JTJ)) + 1e-8 * jnp.eye(6)
        th_new = jnp.clip(th + jnp.linalg.solve(M, J.T @ (x - _sim6(th))),
                          _LOW6, _HIGH6)
        s_new = ssr(th_new)
        ok = s_new < s0
        return (jnp.where(ok, th_new, th),
                jnp.clip(jnp.where(ok, lam * 0.5, lam * 3.0), 1e-7, 1e7),
                jnp.where(ok, s_new, s0)), None

    th0 = _init6(x)
    (th, _, _), _ = lax.scan(body, (th0, jnp.asarray(1e-2), ssr(th0)),
                             None, length=_LM_ITERS)
    return th


def _fisher_logsig(th, sigma):
    """Log per-parameter marginal std from the Laplace covariance F^-1 at the fit,
    F = J^T J / sigma^2. Encodes the target-specific posterior width."""
    J = _jac6(th)
    F = (J.T @ J) / (sigma ** 2 + 1e-24)
    Finv = jnp.linalg.inv(F + 1e-6 * jnp.trace(F) / 6.0 * jnp.eye(6))
    return 0.5 * jnp.log(jnp.clip(jnp.diag(Finv), 1e-24, 1.0))


@jit
def _compress_batch(x):
    th_hat = vmap(_gn_fit)(x)                                        # (n, 6)
    resid = x - vmap(_sim6)(th_hat)
    resid_rms = jnp.sqrt(jnp.mean(resid ** 2, axis=-1, keepdims=True))  # (n, 1)
    logsig = vmap(_fisher_logsig)(th_hat, resid_rms[:, 0])           # (n, 6)
    return jnp.concatenate([th_hat, resid_rms, logsig], axis=-1)


def score_compress(x):
    """Compress raw flux (N_OBS,) or a batch (n, N_OBS) to 13 summaries."""
    x = jnp.asarray(x)
    single = x.ndim == 1
    out = _compress_batch(x[None] if single else x)
    return out[0] if single else out


def simulate_compressed(n_sims):
    """Draw from the prior, simulate noisy curves, return (theta, summaries)."""
    theta, x = simulate_dataset(n_sims, noiseless=False)
    return theta, np.asarray(score_compress(jnp.asarray(x)))