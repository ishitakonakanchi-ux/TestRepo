"""Train a noise-aware amortized transit NPE and save it.

The weighted mode uses 13 summaries (weighted approximate-MLE parameters,
log10 jitter, and weighted Fisher log-widths). The hybrid mode appends the 50
whitened residuals. The full mode also includes the fitted baseline and 50 log
flux errors. Embedded uses those same full inputs with a learned context encoder
that retains the fit summaries as a skip connection. Robust adds a four-start
fit so separated grazing/non-grazing likelihood basins are visible to that
encoder. All modes train only on simulated transit fluxes while sampling
empirical per-bin error profiles from the DR25 library.

Core mode keeps the robust full-context compressor but trains the density
estimator only on ``b``, duration, radius ratio, and ``t0``. Limb darkening and
jitter remain randomized simulator nuisance parameters and are therefore
marginalized rather than predicted. Core-domain additionally augments training
with moderate correlated noise, trends, outliers, error rescaling, and missing
bins while retaining a substantial fraction of clean Gaussian simulations.

Training draws mini-batches from a compressed pool that is regenerated (fresh
simulations, recompressed) every REFRESH_EVERY epochs — a compromise between
recompressing every epoch (costly) and a single frozen pool.

Usage:
    python train_sbi.py [--compression MODE]

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

from npe_wrapper import FullContextEmbedding, NPEEstimator
from transit_sbi import (COMPRESSION_MODES, FULL_LABELS, HYBRID_LABELS,
                         SUMMARY_LABELS, CORE_MODES, CORE_PARAM_INDICES,
                         CORE_PARAM_LABELS, CORE_PRIOR_LOW, CORE_PRIOR_HIGH,
                         DEFAULT_ERROR_LIBRARY, dr25_in_prior_mask,
                         load_flux_err_profiles, simulate_compressed,
                         PARAM_LABELS, PRIOR_LOW, PRIOR_HIGH)

SEED = 42
N_HOLDOUT = 10


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError as err:
        raise SystemExit(f"{name} must be an integer") from err


N_POOL = _env_int("NPE_POOL_SIZE", "150000")
REFRESH_EVERY = _env_int("NPE_REFRESH_EVERY", "50")
N_SIMS_PER_EPOCH = _env_int("NPE_SIMS_PER_EPOCH", "20000")
N_EPOCHS = _env_int("NPE_EPOCHS", "1000")
PATIENCE = _env_int("NPE_PATIENCE", "50")
DEVICE = os.environ.get("NPE_DEVICE", "cpu")  # "cpu", "mps", or "cuda"
CPU_THREADS = _env_int("NPE_CPU_THREADS", "2")


def validate_training_config():
    """Fail before allocating a simulation pool for invalid run settings."""
    values = {
        "NPE_POOL_SIZE": N_POOL,
        "NPE_REFRESH_EVERY": REFRESH_EVERY,
        "NPE_SIMS_PER_EPOCH": N_SIMS_PER_EPOCH,
        "NPE_EPOCHS": N_EPOCHS,
        "NPE_PATIENCE": PATIENCE,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise SystemExit(f"{', '.join(invalid)} must be positive")
    n_validation = max(1, int(N_SIMS_PER_EPOCH * 0.1))
    if n_validation >= N_POOL:
        raise SystemExit(
            "NPE_POOL_SIZE must exceed the fixed validation-set size "
            f"({n_validation})"
        )
    try:
        device = torch.device(DEVICE)
    except RuntimeError as err:
        raise SystemExit(f"Invalid NPE_DEVICE {DEVICE!r}: {err}") from err
    if device.type not in ("cpu", "cuda", "mps"):
        raise SystemExit("NPE_DEVICE must be cpu, cuda, or mps")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("NPE_DEVICE requests CUDA, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("NPE_DEVICE requests MPS, but MPS is unavailable")
    if device.type == "cuda":
        if CPU_THREADS not in (1, 2):
            raise SystemExit("NPE_CPU_THREADS must be 1 or 2 with CUDA")
        torch.set_num_threads(CPU_THREADS)
        torch.set_num_interop_threads(1)


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
        self._closed = False
        self._executor = ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn"),
            initializer=_worker_init)
        # First pool: block (training can't start without it); then prefetch #2.
        try:
            self.theta, self.summ = self._submit_next().result()
            self._future = self._submit_next()
        except BaseException:
            self.close()
            raise

    def _submit_next(self):
        self._gen += 1
        return self._executor.submit(
            _gen_pool, self.pool_size, self.seed + self._gen, self.compression,
            self.flux_err_profiles)

    def __call__(self, n):
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("pool sample size must be a positive integer")
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
        """Cancel prefetched work and shut the worker down promptly.

        ``ProcessPoolExecutor.shutdown(cancel_futures=True)`` does not cancel a
        task that is already running. Without terminating that worker, Python's
        interpreter shutdown waits for an entire unused 150k-curve refresh
        after early stopping and model serialization.
        """
        if self._closed:
            return
        self._closed = True
        process_map = getattr(self._executor, "_processes", None) or {}
        processes = list(process_map.values())
        self._executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compression", choices=COMPRESSION_MODES, default="robust",
        help=("conditioning/target mode; core modes marginalize limb darkening "
              "and jitter"),
    )
    args = parser.parse_args()
    validate_training_config()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    profiles = load_flux_err_profiles()
    if len(profiles) <= N_HOLDOUT:
        raise SystemExit(
            f"Build more than {N_HOLDOUT} Kepler error profiles before training."
        )
    with np.load(DEFAULT_ERROR_LIBRARY) as library:
        eligible = dr25_in_prior_mask(library)
    if np.count_nonzero(eligible) < N_HOLDOUT:
        raise SystemExit(
            f"Only {np.count_nonzero(eligible)} DR25 targets have best-fit b, "
            f"duration, and rp/rs within the NPE prior; need {N_HOLDOUT}."
        )
    order = np.random.default_rng(SEED).permutation(len(profiles))
    holdout_indices = np.sort(order[eligible[order]][:N_HOLDOUT])
    train_indices = np.setdiff1d(np.arange(len(profiles)), holdout_indices)
    train_profiles = profiles[train_indices]
    print(
        f"Error profiles: {len(train_profiles)} train, "
        f"{len(holdout_indices)} held out for PIT"
    )

    cpu_note = f"; {CPU_THREADS} CPU threads" if DEVICE.startswith("cuda") else ""
    print(f"Backend:  flow (torch) -> {DEVICE}{cpu_note}"
          f"   |   compression (jax) -> {jax.default_backend()}")

    print(f"Building compressed pool ({N_POOL} curves, refresh every "
          f"{REFRESH_EVERY} epochs)...")
    simulate_fn = RefreshingPool(
        N_POOL, REFRESH_EVERY, compression=args.compression,
        flux_err_profiles=train_profiles, seed=SEED
    )

    core = args.compression in CORE_MODES
    target_low = CORE_PRIOR_LOW if core else PRIOR_LOW
    target_high = CORE_PRIOR_HIGH if core else PRIOR_HIGH
    prior = BoxUniform(
        low=torch.tensor(target_low), high=torch.tensor(target_high)
    )
    npe = NPEEstimator(
        model="nsf", hidden_features=128, num_transforms=10, learning_rate=1.5e-3,
        batch_size=1024, validation_fraction=0.1,
        embedding_net=(FullContextEmbedding()
                       if args.compression in (
                           "embedded", "robust", "core", "core_domain"
                       ) else None),
        device=DEVICE,
    )
    try:
        npe.fit_online(simulate_fn=simulate_fn, sigma=0.0, prior=prior,
                       n_sims_per_epoch=N_SIMS_PER_EPOCH,
                       n_epochs=N_EPOCHS, patience=PATIENCE)
    finally:
        simulate_fn.close()

    os.makedirs("weights", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm")
    model_fname = f"weights/npe_{args.compression}_{timestamp}.pkl"
    labels = {
        "weighted": SUMMARY_LABELS,
        "hybrid": HYBRID_LABELS,
        "full": FULL_LABELS,
        "embedded": FULL_LABELS,
        "robust": FULL_LABELS,
        "core": FULL_LABELS,
        "core_domain": FULL_LABELS,
    }[args.compression]
    npe.save(model_fname, metadata={
        "schema_version": 5 if core else 4,
        "compression": args.compression,
        "noise_model": ("empirical_errors_plus_marginalized_domain_noise"
                        if args.compression == "core_domain"
                        else "mixed_flux_err_plus_log10_jitter"),
        "prior_low": PRIOR_LOW,
        "prior_high": PRIOR_HIGH,
        "target_prior_low": target_low,
        "target_prior_high": target_high,
        "target_labels": CORE_PARAM_LABELS if core else list(PARAM_LABELS),
        "target_indices": (list(CORE_PARAM_INDICES)
                           if core else list(range(len(PARAM_LABELS)))),
        "summary_labels": labels,
        "error_profile_count": len(profiles),
        "training_profile_count": len(train_profiles),
        "holdout_profile_indices": holdout_indices.tolist(),
        "holdout_selection": "DR25 best-fit b, duration, and rp/rs in prior",
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
