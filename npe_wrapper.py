"""
Sklearn-like wrapper around sbi's NPE inference.

Provides a simple interface to configure, train, and sample from a neural
posterior estimator.

Example
-------
>>> from npe_wrapper import NPEEstimator
>>> import numpy as np
>>> import torch
>>>
>>> prior = torch.distributions.Uniform(torch.zeros(2), torch.ones(2))
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
import torch.nn as nn
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader
from sbi.neural_nets import posterior_nn
from sbi.inference.posteriors import DirectPosterior
from sbi.utils import BoxUniform


class NPEEstimator:
    """Sklearn-like wrapper for Neural Posterior Estimation via sbi.

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
    embedding_net : torch.nn.Module or None
        Optional network that compresses the observation vector before it
        enters the flow. When the observable `x` is high-dimensional (e.g.
        a 200-point light curve), passing the raw vector to the flow is
        inefficient: the flow must learn both the structure of the input
        space and the parameter dependence simultaneously. An embedding
        network first maps x -> z (a lower-dimensional summary), so the
        flow only has to model p(theta | z).

        In practice this can be a simple MLP, e.g.

            embedding_net = torch.nn.Sequential(
                torch.nn.Linear(200, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 8),
            )

        The embedding is trained jointly with the flow, so it learns which
        features of the light curve are informative for the parameters.
        Alternatively, hand-crafted summary statistics (transit depth,
        duration, ingress slope, ...) can be computed before calling fit()
        and passed as a lower-dimensional `x`, avoiding the need for an
        embedding network altogether.

        If None (default), the raw observation is passed directly to the
        flow (equivalent to an identity embedding).
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(self, model="maf", hidden_features=50, num_transforms=5,
                 num_components=6, learning_rate=5e-4,
                 batch_size=50, stop_after_epochs=20,
                 validation_fraction=0.1, clip_max_norm=5.0,
                 embedding_net=None, device="cpu"):
        self.model = model
        self.hidden_features = hidden_features
        self.num_transforms = num_transforms
        self.num_components = num_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.stop_after_epochs = stop_after_epochs
        self.validation_fraction = validation_fraction
        self.clip_max_norm = clip_max_norm
        self.embedding_net = embedding_net
        self.device = device

        self.posterior_ = None
        self.summaries_ = None

    def _build_net(self, theta_sample, x_sample):
        """Construct the neural density estimator.

        Parameters
        ----------
        theta_sample : Tensor
            Sample of parameters for z-scoring.
        x_sample : Tensor
            Sample of observations for z-scoring.

        Returns
        -------
        net : nn.Module
            Neural density estimator.
        """
        kwargs = {
            "model": self.model,
            "hidden_features": self.hidden_features,
            "z_score_x": "independent",
            "z_score_theta": "independent",
        }
        if self.model == "mdn":
            kwargs["num_components"] = self.num_components
        else:
            kwargs["num_transforms"] = self.num_transforms

        if self.embedding_net is not None:
            kwargs["embedding_net"] = self.embedding_net

        build_fn = posterior_nn(**kwargs)
        return build_fn(theta_sample, x_sample)

    def fit_online(self, simulate_fn, sigma, prior, n_sims_per_epoch,
                   n_epochs, patience=20):
        """Train NPE with fresh training data each epoch.

        Each epoch draws entirely new (theta, x) pairs from the simulator,
        giving the network unlimited effective training data. A fixed
        validation set (generated once) provides a stable early-stopping
        signal.

        Parameters
        ----------
        simulate_fn : callable
            Function with signature ``simulate_fn(n_sims) -> (theta, x)``
            where theta is (n_sims, n_params) and x is (n_sims, n_obs).
            Should return *noiseless* data.
        sigma : float
            Gaussian noise standard deviation added to simulations.
        prior : torch Distribution
            Prior distribution over parameters.
        n_sims_per_epoch : int
            Number of fresh simulations drawn each epoch for training.
        n_epochs : int
            Maximum number of training epochs.
        patience : int
            Early stopping patience (epochs without validation improvement).

        Returns
        -------
        self
        """
        # Fixed validation set (generated once, with noise)
        theta_val_np, x_val_np = simulate_fn(
            max(1, int(n_sims_per_epoch * self.validation_fraction)))
        x_val_np = x_val_np + np.random.normal(0, sigma, x_val_np.shape)
        theta_val = torch.as_tensor(
            np.asarray(theta_val_np, dtype=np.float32)).to(self.device)
        x_val = torch.as_tensor(
            np.asarray(x_val_np, dtype=np.float32)).to(self.device)

        # Build the network from an initial batch (for z-scoring)
        theta_init, x_init = simulate_fn(n_sims_per_epoch)
        x_init = x_init + np.random.normal(0, sigma, x_init.shape)
        net = self._build_net(
            torch.as_tensor(np.asarray(theta_init, dtype=np.float32)),
            torch.as_tensor(np.asarray(x_init, dtype=np.float32)),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=patience // 3,
            min_lr=1e-5)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        train_losses, val_losses = [], []

        print(f"Training on device: {next(net.parameters()).device}")
        pbar = trange(n_epochs, desc="Training")
        for epoch in pbar:
            # Fresh training data each epoch
            theta_np, x_np = simulate_fn(n_sims_per_epoch)
            x_np = x_np + np.random.normal(0, sigma, x_np.shape)
            theta_t = torch.as_tensor(
                np.asarray(theta_np, dtype=np.float32))
            x_t = torch.as_tensor(
                np.asarray(x_np, dtype=np.float32))

            train_loader = DataLoader(
                TensorDataset(theta_t, x_t),
                batch_size=self.batch_size, shuffle=True, drop_last=True)

            # Train
            net.train()
            epoch_loss = 0.0
            n_batches = 0
            for tb, xb in train_loader:
                tb, xb = tb.to(self.device), xb.to(self.device)
                optimizer.zero_grad()
                loss = net.loss(tb, xb).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    net.parameters(), self.clip_max_norm)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_losses.append(epoch_loss / max(n_batches, 1))

            # Validate on fixed set
            net.eval()
            with torch.no_grad():
                val_loss = net.loss(theta_val, x_val).mean().item()
            val_losses.append(val_loss)

            scheduler.step(val_loss)
            pbar.set_postfix(
                train=train_losses[-1], val=val_losses[-1],
                lr=optimizer.param_groups[0]["lr"])

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {k: v.clone()
                              for k, v in net.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model and wrap in sbi's DirectPosterior
        net.load_state_dict(best_state)
        net.eval()

        self.posterior_ = DirectPosterior(
            posterior_estimator=net, prior=prior, device=self.device)
        self.summaries_ = [{"training_loss": train_losses,
                            "validation_loss": val_losses}]
        return self

    def sample(self, x_obs, n_samples=10_000, show_progress_bars=True):
        """Draw posterior samples given an observation.

        Parameters
        ----------
        x_obs : array (n_features,)
            Observed data vector.
        n_samples : int
            Number of posterior samples.
        show_progress_bars : bool
            Whether to show sbi's internal progress bars.

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
            (n_samples,), x=x_tensor,
            show_progress_bars=show_progress_bars).numpy()

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
