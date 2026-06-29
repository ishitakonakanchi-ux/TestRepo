import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jaxoplanet.orbits import TransitOrbit
from jaxoplanet.light_curves import limb_dark_light_curve

SIGMA = 5e-4  # Gaussian noise std on flux

# Observation grid: 50 time points in [-0.2, 0.2] days around mid-transit
N_OBS = 50
t_obs = jnp.linspace(-0.2, 0.2, N_OBS)

# ═══════════════════════════════════════════════════════════════════
# Fixed values for parameters we no longer infer:
#   period: cannot be determined from a single phase-folded transit
#   t0:     phase-folding already centers the transit at t=0
# ═══════════════════════════════════════════════════════════════════
DEFAULT_PERIOD = 1.0   # days (any reasonable value works for training)


# Prior bounds for the inferred parameters
#   b        ~ Uniform(0, 0.9)        impact parameter
#   duration ~ Uniform(0.05, 0.35)    transit duration [days]
#   rp_rs    ~ Uniform(0.03, 0.25)    planet-to-star radius ratio
#   u1       ~ Uniform(0, 0.5)        limb-darkening coefficient
#   u2       ~ Uniform(0, 0.5)        limb-darkening coefficient
#   t0       ~ Uniform(-0.05, 0.05)   mid-transit time [days]
#   scatter  ~ Uniform(1e-5, 1e-2)    observational noise level
PRIOR_LOW = [0.0, 0.05, 0.03, 0.0, 0.0, -0.05, 1e-5]
PRIOR_HIGH = [0.9, 0.35, 0.25, 0.5, 0.5, 0.05, 1e-2]
PARAM_LABELS = ["b", "duration", "rp_rs", "u1", "u2", "t0", "scatter"]



@jit
def simulator(params, t_grid=None, period=DEFAULT_PERIOD):
    """Simulate a noiseless transit light curve.

    Parameters
    ----------
    params : array (7,)
        [b, duration, rp_rs, u1, u2, t0, scatter] — the 7 transit parameters
        we infer.
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
    b, duration, rp_rs, u1, u2, t0, scatter = params
    t = t_obs if t_grid is None else t_grid

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


def augment_noise(theta, x_noiseless, n_augmentations):
    """Replicate noiseless simulations with independent noise draws."""
    theta_aug = np.tile(theta, (n_augmentations, 1))
    x_aug = np.tile(x_noiseless, (n_augmentations, 1))
    #using per-row scatter
    scatter_vals = theta_aug[:, -1]
    noise = np.random.normal(0, 1, x_aug.shape) * scatter_vals[:, None]
    x_aug = x_aug + noise
    return theta_aug, x_aug