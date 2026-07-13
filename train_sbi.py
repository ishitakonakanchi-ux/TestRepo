"""Train a noise-aware amortized transit NPE and save it.

The weighted mode uses 13 summaries (weighted approximate-MLE parameters,
log10 jitter, and weighted Fisher log-widths). The hybrid mode appends the 50
whitened residuals. Both modes train only on simulated transit fluxes while
sampling empirical per-bin error profiles from the DR25 library.

Training draws mini-batches from a compressed pool that is regenerated (fresh
simulations, recompressed) every REFRESH_EVERY epochs — a compromise between
recompressing every epoch (costly) and a single frozen pool.

Usage:
    python train_sbi.py [--compression weighted|hybrid]

Outputs:
    weights/npe_<mode>_<timestamp>.pkl   trained posterior
    plots/<mode>_training_loss.png       loss curves
"""
import argparse
import os
import logging
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="sbi")
warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")

import numpy as np
import torch
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sbi.utils import BoxUniform

from npe_wrapper import NPEEstimator
from transit_sbi import (COMPRESSION_MODES, HYBRID_LABELS, SUMMARY_LABELS,
                         load_flux_err_profiles, simulate_compressed,
                         PRIOR_LOW, PRIOR_HIGH)

SEED = 42
N_POOL = 150_000        # curves per pool
REFRESH_EVERY = 50      # regenerate (re-simulate + recompress) the pool every N epochs
DEVICE = "cpu"          # "cpu", "mps", or "cuda" for the flow training


def _worker_init():
    """Runs once in the spawned prefetch worker: drop it to low CPU priority so
    its compression yields to the main flow training. JAX-CPU has no clean thread
    cap on macOS (XLA_FLAGS don't bound its runtime pool, and there's no CPU
    affinity API), so we deprioritise the whole worker instead of pinning cores:
    training wins contention and the prefetch soaks up only the spare cycles,
    which is all it needs given its ~refresh_every-epoch head start."""
    try:
        os.nice(19)
    except OSError:
        pass


def _gen_pool(pool_size, seed, compression, flux_err_profiles):
    """Worker: draw a fresh compressed pool. Runs in a spawned subprocess, so
    its JAX/XLA compression overlaps the main process's flow training."""
    np.random.seed(seed)
    theta, summ = simulate_compressed(
        pool_size, mode=compression, flux_err_profiles=flux_err_profiles
    )
    return np.asarray(theta, np.float32), np.asarray(summ, np.float32)


class RefreshingPool:
    """Yield mini-batches of compressed summaries, resampled from a pool that is
    regenerated with fresh simulations every `refresh_every` epochs. The first
    request is removed from the pool as a disjoint validation set; subsequent
    requests supply network setup and training.

    The next pool is prefetched in a background process so training never stalls
    on regeneration: each swap-in immediately kicks off the following pool, which
    is compressed (JAX, CPU) concurrently with the flow training (torch, CPU).
    The main process runs no JAX itself, keeping its threadpool free for torch."""

    def __init__(self, pool_size, refresh_every, compression,
                 flux_err_profiles, seed=0):
        self.pool_size = pool_size
        self.refresh_every = refresh_every
        self.compression = compression
        self.flux_err_profiles = flux_err_profiles
        self.seed = seed
        self.epoch = -1
        self._gen = 0
        self._executor = ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn"),
            initializer=_worker_init)
        # First pool: block (training can't start without it); then prefetch #2.
        self.theta, self.summ = self._submit_next().result()
        self._future = self._submit_next()

    def _submit_next(self):
        self._gen += 1
        return self._executor.submit(
            _gen_pool, self.pool_size, self.seed + self._gen, self.compression,
            self.flux_err_profiles)

    def __call__(self, n):
        self.epoch += 1
        if self.epoch == 0:
            if n >= len(self.theta):
                raise ValueError("validation set must be smaller than the pool")
            theta, summ = self.theta[:n], self.summ[:n]
            self.theta, self.summ = self.theta[n:], self.summ[n:]
            return theta, summ
        if self.epoch > 0 and self.epoch % self.refresh_every == 0:
            # Swap in the prefetched pool (blocks only if not yet ready), then
            # start generating the one after it.
            self.theta, self.summ = self._future.result()
            self._future = self._submit_next()
        idx = np.random.randint(0, len(self.theta), size=n)
        return self.theta[idx], self.summ[idx]

    def close(self):
        """Cancel any pending prefetch and shut the worker down."""
        self._executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compression", choices=COMPRESSION_MODES, default="weighted",
        help="weighted summaries, or weighted summaries plus whitened residuals",
    )
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    profiles = load_flux_err_profiles()
    if len(profiles) < 5:
        raise SystemExit("Build at least 5 Kepler error profiles before training.")
    order = np.random.default_rng(SEED).permutation(len(profiles))
    n_holdout = max(1, len(profiles) // 5)
    holdout_indices = np.sort(order[:n_holdout])
    train_indices = np.sort(order[n_holdout:])
    train_profiles = profiles[train_indices]
    print(
        f"Error profiles: {len(train_profiles)} train, "
        f"{len(holdout_indices)} held out for PIT"
    )

    mps_ok = torch.backends.mps.is_available()
    print(f"Backend:  flow (torch) -> {DEVICE}"
          f"{'' if DEVICE != 'mps' or mps_ok else ' [WARNING: MPS unavailable]'}"
          f"   |   compression (jax) -> {jax.default_backend()}")

    print(f"Building compressed pool ({N_POOL} curves, refresh every "
          f"{REFRESH_EVERY} epochs)...")
    simulate_fn = RefreshingPool(
        N_POOL, REFRESH_EVERY, compression=args.compression,
        flux_err_profiles=train_profiles, seed=SEED
    )

    prior = BoxUniform(low=torch.tensor(PRIOR_LOW), high=torch.tensor(PRIOR_HIGH))
    npe = NPEEstimator(
        model="nsf", hidden_features=128, num_transforms=10, learning_rate=1.5e-3,
        batch_size=1024, validation_fraction=0.1,
        embedding_net=None, device=DEVICE,
    )
    npe.fit_online(simulate_fn=simulate_fn, sigma=0.0, prior=prior,
                   n_sims_per_epoch=20_000, n_epochs=1000, patience=50)
    simulate_fn.close()

    os.makedirs("weights", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm")
    model_fname = f"weights/npe_{args.compression}_{timestamp}.pkl"
    labels = SUMMARY_LABELS if args.compression == "weighted" else HYBRID_LABELS
    npe.save(model_fname, metadata={
        "schema_version": 4,
        "compression": args.compression,
        "noise_model": "mixed_flux_err_plus_log10_jitter",
        "prior_low": PRIOR_LOW,
        "prior_high": PRIOR_HIGH,
        "summary_labels": labels,
        "error_profile_count": len(profiles),
        "training_profile_count": len(train_profiles),
        "holdout_profile_indices": holdout_indices.tolist(),
    })
    print(f"Saved {os.path.abspath(model_fname)}")

    summary = npe.summaries_[0]
    fig, ax = plt.subplots()
    ax.plot(summary["training_loss"], label="Training")
    ax.plot(summary["validation_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    loss_fname = f"plots/{args.compression}_training_loss.png"
    fig.savefig(loss_fname, dpi=150, bbox_inches="tight")
    print(f"Saved {loss_fname}")
