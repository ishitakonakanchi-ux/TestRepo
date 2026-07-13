"""
Sklearn-like wrapper around sbi's NPE inference.

Provides a simple interface to configure, train, and sample from a neural
posterior estimator.

Example
-------
>>> from npe_wrapper import NPEEstimator
>>> import numpy as np
>>> import torch
>>> from sbi.utils import BoxUniform
>>>
>>> prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))
>>> def simulate(n):                         # (n_sims) -> (theta, x)
...     th = np.random.rand(n, 2)
...     return th, np.column_stack([th[:, 0]**2, np.sin(th[:, 1])])
>>> npe = NPEEstimator(model="nsf", hidden_features=50, num_transforms=5)
>>> npe.fit_online(simulate, sigma=0.01, prior=prior,
...                n_sims_per_epoch=1000, n_epochs=100)
>>> samples = npe.sample(np.array([0.1, 0.2]), n_samples=5000)  # (n_samples, n_params)
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


def _prior_to_cpu(prior):
    """Return a CPU copy of a BoxUniform / Independent(Uniform) prior.

    sbi's ``BoxUniform`` is ``Independent(Uniform(low, high))``. Reconstructing
    it on CPU is cheaper and safer than mutating the live distribution.
    Falls back to the original prior if its structure is unrecognised.
    """
    base = getattr(prior, "base_dist", None)
    low = getattr(base, "low", None)
    high = getattr(base, "high", None)
    if low is None or high is None:
        return prior
    return BoxUniform(low=low.detach().cpu(), high=high.detach().cpu())


def _to_unbounded(theta, lo, hi):
    """Map bounded theta in [lo, hi] to unbounded z via the logit transform."""
    u = ((theta - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6)
    return torch.log(u) - torch.log1p(-u)


def _to_bounded(z, lo, hi):
    """Inverse of _to_unbounded: map z in R back into [lo, hi]."""
    return lo + (hi - lo) * torch.sigmoid(z)


def _sample_on_cpu(posterior, n_samples, x_tensor_cpu, show_progress_bars,
                   max_sampling_time=None):
    """Sample from a DirectPosterior with everything pinned to CPU.

    Avoids an sbi bug where rejection sampling against the prior's support
    raises ``Expected all tensors to be on the same device`` when the
    posterior was trained on MPS. Restores the original device on exit so
    later calls (e.g. log_prob on GPU) are unaffected.

    ``max_sampling_time`` caps the rejection-sampling wall-clock (seconds) and
    returns whatever was collected — a guard for targets whose flow leaks mass
    outside the prior (very low acceptance).
    """
    net = posterior.posterior_estimator
    orig_device = next(net.parameters()).device
    orig_prior = posterior.prior
    orig_post_device = getattr(posterior, "_device", None)

    net.to("cpu")
    posterior.prior = _prior_to_cpu(orig_prior)
    if orig_post_device is not None:
        posterior._device = "cpu"
    extra = ({} if max_sampling_time is None else
             {"max_sampling_time": max_sampling_time,
              "return_partial_on_timeout": True})
    try:
        out = posterior.sample(
            (n_samples,), x=x_tensor_cpu,
            show_progress_bars=show_progress_bars, **extra).cpu().numpy()
    finally:
        net.to(orig_device)
        posterior.prior = orig_prior
        if orig_post_device is not None:
            posterior._device = orig_post_device
    return out


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
        duration, ingress slope, ...) can be computed before calling fit_online()
        and passed as a lower-dimensional `x`, avoiding the need for an
        embedding network altogether.

        If None (default), the raw observation is passed directly to the
        flow (equivalent to an identity embedding).
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(self, model="maf", hidden_features=50, num_transforms=5,
                 num_components=6, learning_rate=5e-4,
                 batch_size=50, validation_fraction=0.1, clip_max_norm=5.0,
                 embedding_net=None, device="cpu"):
        self.model = model
        self.hidden_features = hidden_features
        self.num_transforms = num_transforms
        self.num_components = num_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.clip_max_norm = clip_max_norm
        self.embedding_net = embedding_net
        self.device = device

        self.posterior_ = None
        self.posteriors_ = []  # For ensemble
        self.summaries_ = None
        self.bounds_ = None  # (low, high) if trained in unbounded logit space
        self.metadata_ = {}

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
        # If the prior is a box, learn in an unbounded (logit) space so the flow
        # cannot place mass outside the prior -> sampling needs no rejection.
        base = getattr(prior, "base_dist", None)
        if base is not None and hasattr(base, "low"):
            self.bounds_ = (base.low.detach().cpu().float(),
                            base.high.detach().cpu().float())
        else:
            self.bounds_ = None

        def _tf(theta_t):
            return (theta_t if self.bounds_ is None
                    else _to_unbounded(theta_t, *self.bounds_))

        # Fixed validation set (generated once, with noise)
        theta_val_np, x_val_np = simulate_fn(
            max(1, int(n_sims_per_epoch * self.validation_fraction)))
        x_val_np = x_val_np + np.random.normal(0, sigma, x_val_np.shape)
        theta_val = _tf(torch.as_tensor(
            np.asarray(theta_val_np, dtype=np.float32))).to(self.device)
        x_val = torch.as_tensor(
            np.asarray(x_val_np, dtype=np.float32)).to(self.device)

        # Build the network from an initial batch (for z-scoring)
        theta_init, x_init = simulate_fn(n_sims_per_epoch)
        x_init = x_init + np.random.normal(0, sigma, x_init.shape)
        net = self._build_net(
            _tf(torch.as_tensor(np.asarray(theta_init, dtype=np.float32))),
            torch.as_tensor(np.asarray(x_init, dtype=np.float32)),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=patience // 3,
            min_lr=1e-4)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        train_losses, val_losses = [], []

        print(f"Training on device: {next(net.parameters()).device}")
        pbar = trange(n_epochs, desc="Training")
        for epoch in pbar:
            # Fresh training data each epoch
            theta_np, x_np = simulate_fn(n_sims_per_epoch)
            x_np = x_np + np.random.normal(0, sigma, x_np.shape)
            theta_t = _tf(torch.as_tensor(
                np.asarray(theta_np, dtype=np.float32)))
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
                # Skip non-finite batches so one bad step can't NaN the weights
                # (flows on very sharp conditionals can produce inf loss/grads).
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(
                    net.parameters(), self.clip_max_norm)
                if not torch.isfinite(gnorm):
                    optimizer.zero_grad()
                    continue
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

            # Early stopping (require meaningful improvement)
            min_delta = 1e-4
            if val_loss < best_val_loss - min_delta:
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

    def fit_online_ensemble(self, simulate_fn, sigma, prior, n_sims_per_epoch,
                            n_epochs, patience=20, n_ensemble=3, base_seed=42):
        """Train an ensemble of NPEs with different random seeds.

        Parameters
        ----------
        simulate_fn, sigma, prior, n_sims_per_epoch, n_epochs, patience :
            Same as fit_online.
        n_ensemble : int
            Number of networks to train.
        base_seed : int
            Base random seed (each network uses base_seed + i).

        Returns
        -------
        self
        """
        import copy
        self.posteriors_ = []
        self.summaries_ = []

        for i in range(n_ensemble):
            print(f"\n=== Training ensemble member {i+1}/{n_ensemble} ===")
            seed = base_seed + i
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Create fresh embedding net for each member
            embedding_copy = copy.deepcopy(self.embedding_net)

            # Temporarily swap and train
            original_embedding = self.embedding_net
            self.embedding_net = embedding_copy
            self.fit_online(simulate_fn, sigma, prior, n_sims_per_epoch,
                           n_epochs, patience)
            self.embedding_net = original_embedding

            # Store results
            self.posteriors_.append(self.posterior_)
            self.summaries_.extend(self.summaries_)
            self.posterior_ = None  # Clear single posterior

        print(f"\nEnsemble training complete: {n_ensemble} networks")
        return self

    def sample(self, x_obs, n_samples=10_000, show_progress_bars=True,
               max_sampling_time=None):
        """Draw posterior samples given an observation.

        Parameters
        ----------
        x_obs : array (n_features,)
            Observed data vector.
        n_samples : int
            Number of posterior samples.
        show_progress_bars : bool
            Whether to show sbi's internal progress bars.
        max_sampling_time : float or None
            Cap on rejection-sampling wall-clock in seconds; returns whatever
            was collected. Guards against targets whose flow leaks mass outside
            the prior (very low acceptance -> otherwise very slow).

        Returns
        -------
        samples : array (n_samples, n_params)
            Posterior samples.
        """
        # Sampling is forced onto CPU. sbi's DirectPosterior.sample uses
        # rejection sampling against the prior's support; on MPS this triggers
        # a device-mismatch (mps:0 vs cpu) inside torch's constraint check
        # because sbi mixes CPU and device tensors internally. CPU sampling
        # avoids this and is fast for the small flows used here.
        x_tensor = torch.as_tensor(
            np.array(x_obs, dtype=np.float32, copy=True)).unsqueeze(0).cpu()

        # Unbounded-transform path: the flow was trained in logit space, so
        # sample from it directly (no rejection) and map back into the box.
        # Every sample is in-prior by construction -> 100% acceptance.
        if self.bounds_ is not None and self.posterior_ is not None:
            de = self.posterior_.posterior_estimator
            orig_device = next(de.parameters()).device
            de.to("cpu")
            try:
                with torch.no_grad():
                    z = de.net.sample(n_samples, context=x_tensor)
                z = z.reshape(-1, z.shape[-1])
                theta = _to_bounded(z, *self.bounds_).cpu().numpy()
            finally:
                de.to(orig_device)
            return theta[:n_samples]

        # Ensemble: sample equally from each posterior
        if self.posteriors_:
            samples_per = n_samples // len(self.posteriors_)
            all_samples = []
            for post in self.posteriors_:
                s = _sample_on_cpu(post, samples_per, x_tensor,
                                   show_progress_bars, max_sampling_time)
                all_samples.append(s)
            return np.concatenate(all_samples, axis=0)

        # Single posterior
        if self.posterior_ is None:
            raise RuntimeError("Call fit_online() before sample().")
        return _sample_on_cpu(self.posterior_, n_samples, x_tensor,
                              show_progress_bars, max_sampling_time)

    def sample_batch(self, x_obs, n_samples=1000):
        """Sample the posterior for a *batch* of observations in one vectorised
        flow pass — far faster than looping ``sample`` when many posteriors are
        needed at once (e.g. PIT / calibration).

        Parameters
        ----------
        x_obs : array (n_obs, n_features)
            Batch of observation vectors.
        n_samples : int
            Posterior samples per observation.

        Returns
        -------
        samples : array (n_obs, n_samples, n_params)
        """
        if self.posterior_ is None:
            raise RuntimeError("Call fit_online() before sample_batch().")
        x = torch.as_tensor(np.array(x_obs, dtype=np.float32)).cpu()
        de = self.posterior_.posterior_estimator
        orig_device = next(de.parameters()).device
        de.to("cpu")
        try:
            with torch.no_grad():
                # nflows: context (n_obs, F) -> samples (n_obs, n_samples, D).
                z = de.net.sample(n_samples, context=x)
        finally:
            de.to(orig_device)
        if self.bounds_ is not None:
            z = _to_bounded(z, *self.bounds_)
        return z.cpu().numpy()

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
        if self.posterior_ is None and not self.posteriors_:
            raise RuntimeError("Call fit_online() before log_prob().")
        x_tensor = torch.as_tensor(
            np.asarray(x_obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        theta_tensor = torch.as_tensor(
            np.asarray(theta, dtype=np.float32)).to(self.device)

        if self.posteriors_:
            # Average log probs across ensemble
            log_probs = []
            for post in self.posteriors_:
                lp = post.log_prob(theta_tensor, x=x_tensor).cpu().numpy()
                log_probs.append(lp)
            return np.mean(log_probs, axis=0)

        return self.posterior_.log_prob(
            theta_tensor, x=x_tensor).cpu().numpy()

    def save(self, path, metadata=None):
        """Save the trained posterior(s) to a pickle file.

        Parameters
        ----------
        path : str
            Output file path.
        metadata : dict or None
            Small preprocessing manifest stored alongside the posterior.
        """
        if self.posteriors_:
            with open(path, "wb") as f:
                pickle.dump(self.posteriors_, f)
        elif self.posterior_ is not None:
            self.metadata_ = metadata or {}
            with open(path, "wb") as f:
                pickle.dump({"posterior": self.posterior_,
                             "bounds": self.bounds_,
                             "metadata": self.metadata_}, f)
        else:
            raise RuntimeError("Call fit_online() before save().")

    def load(self, path):
        """Load previously trained posterior(s) from a pickle file.

        Parameters
        ----------
        path : str
            Path to the saved posterior.

        Returns
        -------
        self
        """
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
        except RuntimeError:
            # Model saved on a device (MPS/CUDA) unavailable here -> remap to CPU.
            import io
            orig = torch.storage._load_from_bytes
            torch.storage._load_from_bytes = lambda b: torch.load(
                io.BytesIO(b), map_location="cpu")
            try:
                with open(path, "rb") as f:
                    loaded = pickle.load(f)
            finally:
                torch.storage._load_from_bytes = orig
        if isinstance(loaded, list):
            self.posteriors_ = loaded
            # Get device from first posterior's neural net
            self.device = str(next(
                self.posteriors_[0].posterior_estimator.parameters()).device)
        else:
            if isinstance(loaded, dict):  # new format: posterior + bounds
                self.posterior_ = loaded["posterior"]
                self.bounds_ = loaded["bounds"]
                self.metadata_ = loaded.get("metadata", {})
            else:                          # old format: bare posterior
                self.posterior_ = loaded
                self.bounds_ = None
                self.metadata_ = {}
            self.device = str(next(
                self.posterior_.posterior_estimator.parameters()).device)
        return self
