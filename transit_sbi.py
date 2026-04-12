import numpy as np
import jax.numpy as jnp

from jax import jit
from jaxoplanet.orbits import TransitOrbit
from jaxoplanet.light_curves import limb_dark_light_curve
from ili.utils import Uniform

from npe_wrapper import NPEEstimator

# Fixed transit parameters
TRUE_PERIOD = 3.0           # orbital period [days]
TRUE_T0 = 0.0               # mid-transit time [days]
TRUE_U = [0.3, 0.2]         # quadratic limb darkening coefficients
SIGMA = 5e-4                # Gaussian noise std on flux

# Observation grid: 200 time points in [-0.2, 0.2] days around mid-transit
N_OBS = 200
t_obs = jnp.linspace(-0.2, 0.2, N_OBS)

# Prior bounds for the 3 inferred parameters
#   b       ~ Uniform(0, 1)      impact parameter [dimensionless]
#   duration ~ Uniform(0.01, 0.5) transit duration [days]
#   rp_rs   ~ Uniform(0.01, 0.3) planet-to-star radius ratio [dimensionless]
PRIOR_LOW = [0.0, 0.01, 0.01]
PRIOR_HIGH = [1.0, 0.5, 0.3]
PARAM_LABELS = ["b", "duration", "rp_rs"]

@jit
def simulator(params):
    """Simulate a noisy transit light curve.

    Parameters
    ----------
    params : array (3,)
        [b, duration, rp_rs] — impact parameter, transit duration [days],
        and planet-to-star radius ratio.

    Returns
    -------
    flux : array (N_OBS,)
        Noisy relative flux evaluated at `t_obs`.
    """
    b, duration, rp_rs = params

    orbit = TransitOrbit(
        period=TRUE_PERIOD,
        duration=duration,
        time_transit=TRUE_T0,
        impact_param=b,
        radius_ratio=rp_rs,
    )
    flux = 1.0 + limb_dark_light_curve(orbit, TRUE_U)(t_obs)
    flux = flux + np.random.normal(0, SIGMA, N_OBS)
    return np.array(flux)


def simulate_dataset(n_sims):
    """Draw parameters from the prior and simulate light curves.

    Parameters
    ----------
    n_sims : int
        Number of simulations.

    Returns
    -------
    theta : array (n_sims, 3)
        Sampled parameters [b, duration, rp_rs].
    x : array (n_sims, N_OBS)
        Simulated noisy light curves.
    """
    theta = np.column_stack([
        np.random.uniform(PRIOR_LOW[i], PRIOR_HIGH[i], n_sims)
        for i in range(3)
    ])
    x = np.array([simulator(t) for t in theta])
    return theta, x


if __name__ == "__main__":
    n_sims = 500

    # Simulate training data: theta (500, 3), x (500, 200)
    theta, x = simulate_dataset(n_sims)

    prior = Uniform(low=PRIOR_LOW, high=PRIOR_HIGH, device="cpu")

    # Train NPE
    npe = NPEEstimator(
        model="maf",
        hidden_features=50,
        num_transforms=5,
        learning_rate=5e-4,
        batch_size=50,
        stop_after_epochs=20,
    )
    npe.fit(theta, x, prior)

    # Sample posterior for the first simulated observation
    # x_obs: (200,) -> samples: (10000, 3)
    samples = npe.sample(x[0], n_samples=10_000)
    print(f"Posterior samples shape: {samples.shape}")
    print(f"True params: {theta[0]}")
    print(f"Posterior mean: {samples.mean(axis=0)}")
