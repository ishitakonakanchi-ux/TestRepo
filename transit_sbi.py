import os
import argparse
import numpy as np
import jax.numpy as jnp #added import
from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.validation import ValidationRunner
from jaxoplanet.orbits import TransitOrbit #added import
from jaxoplanet.light_curves import limb_dark_light_curve #added import

TRUE_PERIOD = 3.0           # orbital period [days] (fixed) 
TRUE_T0 = 0.0              # mid-transit time [days] (fixed)
TRUE_U = [0.3, 0.2]        # quadratic limb darkening coefficients (fixed)
SIGMA = 5e-4  
#above variables taken from TransitMCMC1 directly

num_sims = 500 # changed from 200, previous toy value
t_obs = jnp.linspace(-0.2, 0.2, 200) #TransitMCMC1


def simulator(params):
    # create toy simulations
    #x = np.arange(10)
    #y = 3 * params[0] * np.sin(x) + params[1] * x ** 2 - 2 * params[2] * x
    #y += np.random.randn(len(x))
    #return y
    b = params[0]
    duration = params[1]
    rp_rs = params[2]
    
    orbit = TransitOrbit(
        period=TRUE_PERIOD,
        duration=duration,
        time_transit=TRUE_T0,
        impact_param=b,
        radius_ratio=rp_rs,)
    flux = 1.0 + limb_dark_light_curve(orbit, TRUE_U)(t_obs) #t_obs instead of t
    flux = np.random.normal(0, SIGMA, len(t_obs)) + flux # noise
    return np.array(flux)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run SBI inference for transit data.")
    parser.add_argument(
        "--model", type=str,
        default="NPE",
        help="Configuration file to use for model training.")
    args = parser.parse_args()

    # construct a working directory
    if not os.path.isdir("transit"):
        os.mkdir("transit")

    # simulate data and save as numpy files
    #theta = np.random.rand(200, 3)  # 200 simulations, 3 parameters
    theta = np.array([np.random.uniform(0.0,  1.0,  num_sims), #b
        np.random.uniform(0.01, 0.5,  num_sims), #duration
        np.random.uniform(0.01, 0.3,  num_sims)#rp_rs
    ]).T
    x = np.array([simulator(t) for t in theta])
    np.save("transit/theta.npy", theta)
    np.save("transit/x.npy", x)

    # reload all simulator examples as a dataloader
    all_loader = StaticNumpyLoader.from_config("configs/data/transit.yaml")

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    runner = InferenceRunner.from_config(
        f"configs/infer/transit_sbi_{args.model}.yaml")
    runner(loader=all_loader)

    # use the trained posterior model to predict on a single example from
    # the test set
    val_runner = ValidationRunner.from_config(
        f"configs/val/transit_sbi_{args.model}.yaml")
    val_runner(loader=all_loader)
