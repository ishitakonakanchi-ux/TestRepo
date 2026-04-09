"""
Sklearn-like wrapper around ltu-ili's NPE inference.

Provides a simple interface to configure, train, and sample from a neural
posterior estimator without dealing with YAML configs.

Example
-------
>>> from npe_wrapper import NPEEstimator
>>> import numpy as np
>>> from ili.utils import Uniform
>>>
>>> prior = Uniform(low=[0, 0], high=[1, 1])
>>> theta = np.random.rand(500, 2)          # (n_sims, n_params)
>>> x = np.column_stack([theta[:, 0]**2,    # (n_sims, n_features)
...                      np.sin(theta[:, 1])])
>>> npe = NPEEstimator(model="maf", hidden_features=50, num_transforms=5)
>>> npe.fit(theta, x, prior)
>>> samples = npe.sample(x[0], n_samples=5000)  # (n_samples, n_params)
"""

import pickle

import numpy as np
import torch
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.utils import load_nde_sbi, Uniform


class NPEEstimator:
    """Sklearn-like wrapper for Neural Posterior Estimation via ltu-ili.

    Parameters
    ----------
    model : str
        Density estimator architecture: 'maf', 'nsf', 'made', or 'mdn'.
    hidden_features : int
        Number of hidden units per layer.
    num_transforms : int
        Number of flow transforms (ignored for 'mdn').
    num_components : int
        Number of mixture components (only used for 'mdn').
    ensemble_repeats : int
        Number of identical networks in the ensemble.
    learning_rate : float
        Adam learning rate.
    batch_size : int
        Training mini-batch size.
    stop_after_epochs : int
        Early-stopping patience (epochs without validation improvement).
    validation_fraction : float
        Fraction of data held out for validation.
    clip_max_norm : float
        Gradient clipping max norm.
    device : str
        'cpu' or 'cuda'.
    out_dir : str or None
        Directory to save the trained posterior. None to skip saving.
    """

    def __init__(self, model="maf", hidden_features=50, num_transforms=5,
                 num_components=6, ensemble_repeats=1, learning_rate=5e-4,
                 batch_size=50, stop_after_epochs=20,
                 validation_fraction=0.1, clip_max_norm=5.0,
                 device="cpu", out_dir=None):
        self.model = model
        self.hidden_features = hidden_features
        self.num_transforms = num_transforms
        self.num_components = num_components
        self.ensemble_repeats = ensemble_repeats
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.stop_after_epochs = stop_after_epochs
        self.validation_fraction = validation_fraction
        self.clip_max_norm = clip_max_norm
        self.device = device
        self.out_dir = out_dir

        self.posterior_ = None
        self.summaries_ = None

    def _build_nets(self):
        """Construct the neural density estimator(s).

        Returns
        -------
        nets : list
            List of network builders for the SBI runner.
        """
        kwargs = {"engine": "NPE", "model": self.model,
                  "hidden_features": self.hidden_features,
                  "repeats": self.ensemble_repeats}
        if self.model == "mdn":
            kwargs["num_components"] = self.num_components
        else:
            kwargs["num_transforms"] = self.num_transforms
        return [load_nde_sbi(**kwargs)]

    def _build_train_args(self):
        """Construct the training hyperparameters dict.

        Returns
        -------
        train_args : dict
            Training configuration passed to sbi's `.train()`.
        """
        return {
            "training_batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "validation_fraction": self.validation_fraction,
            "stop_after_epochs": self.stop_after_epochs,
            "clip_max_norm": self.clip_max_norm,
        }

    def fit(self, theta, x, prior):
        """Train the NPE on simulated data.

        Parameters
        ----------
        theta : array (n_sims, n_params)
            Parameter samples drawn from the prior.
        x : array (n_sims, n_features)
            Corresponding simulated observables.
        prior : torch Distribution or ili.utils.Uniform
            Prior distribution over parameters.

        Returns
        -------
        self
        """
        loader = NumpyLoader(x=np.asarray(x), theta=np.asarray(theta))
        runner = InferenceRunner.load(
            backend="sbi", engine="NPE", prior=prior,
            nets=self._build_nets(), train_args=self._build_train_args(),
            device=self.device, out_dir=self.out_dir,
        )
        self.posterior_, self.summaries_ = runner(loader=loader)
        return self

    def sample(self, x_obs, n_samples=10_000):
        """Draw posterior samples given an observation.

        Parameters
        ----------
        x_obs : array (n_features,)
            Observed data vector.
        n_samples : int
            Number of posterior samples.

        Returns
        -------
        samples : array (n_samples, n_params)
            Posterior samples.
        """
        if self.posterior_ is None:
            raise RuntimeError("Call fit() before sample().")
        x_tensor = torch.as_tensor(
            np.asarray(x_obs, dtype=np.float32)).unsqueeze(0)
        return self.posterior_.sample(
            (n_samples,), x=x_tensor).numpy()

    def log_prob(self, theta, x_obs):
        """Evaluate log-posterior at given parameter values.

        Parameters
        ----------
        theta : array (n_eval, n_params)
            Parameter values to evaluate.
        x_obs : array (n_features,)
            Observed data vector.

        Returns
        -------
        log_p : array (n_eval,)
            Log-posterior values.
        """
        if self.posterior_ is None:
            raise RuntimeError("Call fit() before log_prob().")
        x_tensor = torch.as_tensor(
            np.asarray(x_obs, dtype=np.float32)).unsqueeze(0)
        theta_tensor = torch.as_tensor(
            np.asarray(theta, dtype=np.float32))
        return self.posterior_.log_prob(
            theta_tensor, x=x_tensor).numpy()

    def save(self, path):
        """Save the trained posterior to a pickle file.

        Parameters
        ----------
        path : str
            Output file path.
        """
        if self.posterior_ is None:
            raise RuntimeError("Call fit() before save().")
        with open(path, "wb") as f:
            pickle.dump(self.posterior_, f)

    def load(self, path):
        """Load a previously trained posterior from a pickle file.

        Parameters
        ----------
        path : str
            Path to the saved posterior.

        Returns
        -------
        self
        """
        with open(path, "rb") as f:
            self.posterior_ = pickle.load(f)
        return self
