import numpy as np
import jax.numpy as jnp

from jax import jit, vmap
from jaxoplanet.orbits import TransitOrbit
from jaxoplanet.light_curves import limb_dark_light_curve

# NO Fixed transit parameters
#TRUE_PERIOD = 3.0           # orbital period [days]
#TRUE_T0 = 0.0               # mid-transit time [days]
#TRUE_U = [0.3, 0.2]         # quadratic limb darkening coefficients
SIGMA = 5e-4                # Gaussian noise std on flux

# Observation grid: 50 time points in [-0.2, 0.2] days around mid-transit
N_OBS = 50
t_obs = jnp.linspace(-0.2, 0.2, N_OBS)

# Prior bounds for the 3 inferred parameters
#   b       ~ Uniform(0, 1)      impact parameter [dimensionless]
#   duration ~ Uniform(0.01, 0.5) transit duration [days]
#   rp_rs   ~ Uniform(0.01, 0.3) planet-to-star radius ratio [dimensionless]
PRIOR_LOW = [0.0, 0.05, 0.03, 2.0, -0.15, 0.0, 0.0]
PRIOR_HIGH = [0.9, 0.35, 0.25, 4.0, 0.15, 0.5, 0.5]
PARAM_LABELS = ["b", "duration", "rp_rs", "period", "t0", "u1", "u2"]

@jit
def simulator(params):
    """Simulate a noiseless transit light curve.

    Parameters
    ----------
    params : array (3,) 
        [b, duration, rp_rs] — impact parameter, transit duration [days],
        and planet-to-star radius ratio.

    Returns
    -------
    flux : array (N_OBS,)
        Noiseless relative flux evaluated at `t_obs`.
    """
    b, duration, rp_rs, period, t0, u1, u2 = params

    orbit = TransitOrbit(
        period=period,
        duration=duration,
        time_transit=t0,
        impact_param=b,
        radius_ratio=rp_rs,
    )
    return 1.0 + limb_dark_light_curve(orbit, [u1, u2])(t_obs)


simulator_batch = jit(vmap(simulator))


def simulate_dataset(n_sims, noiseless=False):
    """Draw parameters from the prior and simulate light curves.

    Parameters
    ----------
    n_sims : int
        Number of simulations.
    noiseless : bool
        If True, return noiseless light curves (for use with on-the-fly
        noise augmentation).

    Returns
    -------
    theta : array (n_sims, 3)
        Sampled parameters [b, duration, rp_rs].
    x : array (n_sims, N_OBS)
        Simulated light curves (noiseless if requested).
    """
    theta = np.column_stack([
        np.random.uniform(PRIOR_LOW[i], PRIOR_HIGH[i], n_sims)
        for i in range(7)
    ])
    x = np.asarray(simulator_batch(jnp.array(theta)))
    if not noiseless:
        x = x + np.random.normal(0, SIGMA, x.shape)
    return theta, x


def augment_noise(theta, x_noiseless, sigma, n_augmentations):
    """Replicate noiseless simulations with independent noise draws.

    Mimics on-the-fly noise augmentation (Zhang+2020) within sbi's
    constraint of loading all data at once: each noiseless light curve
    is replicated `n_augmentations` times with a fresh noise realization,
    effectively multiplying the training set size.

    Parameters
    ----------
    theta : array (n_sims, n_params)
        Original parameter samples.
    x_noiseless : array (n_sims, n_obs)
        Noiseless light curves.
    sigma : float
        Gaussian noise standard deviation.
    n_augmentations : int
        Number of noise realizations per simulation.

    Returns
    -------
    theta_aug : array (n_sims * n_augmentations, n_params)
        Tiled parameters.
    x_aug : array (n_sims * n_augmentations, n_obs)
        Noisy light curves with independent noise draws.
    """
    theta_aug = np.tile(theta, (n_augmentations, 1))
    x_aug = np.tile(x_noiseless, (n_augmentations, 1))
    x_aug = x_aug + np.random.normal(0, sigma, x_aug.shape)
    return theta_aug, x_aug
