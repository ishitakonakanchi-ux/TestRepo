"""
Train an NPE on transit light curves and save the model.

Usage:
    python train_sbi.py

Outputs (in weights/):
    - npe_<timestamp>.pkl: trained posterior
    - summary_<timestamp>.pkl: training/validation loss curves
Outputs (in plots/):
    - training_loss.png: loss curves
    - pit_calibration.png: PIT histograms
"""

import os
import pickle
import logging
import warnings
from datetime import datetime

logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="sbi")
warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
from sbi.utils import BoxUniform
from npe_wrapper import NPEEstimator
from transit_sbi import simulate_dataset, PRIOR_LOW, PRIOR_HIGH, SIGMA, PARAM_LABELS

# Auto-detect device: CUDA > MPS > CPU
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
SEED = 42
N_ENSEMBLE = 1  # Set > 1 for ensemble training


class CNNEmbedding(nn.Module):
    """1D CNN to compress light curves into a low-dimensional summary.

    Uses larger kernels and dilated convolutions to capture the full
    transit shape, which helps recover the impact parameter.
    """
    def __init__(self, output_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            # Large kernel to capture global transit shape
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(),
            # Dilated conv for larger receptive field without more params
            nn.Conv1d(32, 64, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(10),  # 10 divides 50 (N_OBS)
        )
        self.fc = nn.Sequential(
            nn.Linear(640, 128), nn.ReLU(),  # 64 channels * 10 = 640
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    prior = BoxUniform(low=torch.tensor(PRIOR_LOW), high=torch.tensor(PRIOR_HIGH),
                        device=DEVICE)
    embedding_net = CNNEmbedding(output_dim=32)

    npe = NPEEstimator(
        model="maf",
        hidden_features=256,
        num_transforms=10,
        learning_rate=1e-3,
        batch_size=512,
        stop_after_epochs=50,
        validation_fraction=0.3,
        embedding_net=embedding_net,
        device=DEVICE,
    )
    if N_ENSEMBLE > 1:
        npe.fit_online_ensemble(
            simulate_fn=lambda n: simulate_dataset(n, noiseless=True),
            sigma=SIGMA,
            prior=prior,
            n_sims_per_epoch=10000,
            n_epochs=2000,
            patience=100,
            n_ensemble=N_ENSEMBLE,
            base_seed=SEED,
        )
    else:
        npe.fit_online(
            simulate_fn=lambda n: simulate_dataset(n, noiseless=True),
            sigma=SIGMA,
            prior=prior,
            n_sims_per_epoch=10000,
            n_epochs=2000,
            patience=100,
        )

    # Save model and training summary
    os.makedirs("weights", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm")
    model_fname = f"weights/npe_{timestamp}.pkl"
    summary_fname = f"weights/summary_{timestamp}.pkl"

    npe.save(model_fname)
    with open(summary_fname, "wb") as f:
        pickle.dump(npe.summaries_, f)

    print(f"Saved {os.path.abspath(model_fname)}")
    print(f"Saved {os.path.abspath(summary_fname)}")

    # Plot training loss
    summary = npe.summaries_[0]
    fig, ax = plt.subplots()
    ax.plot(summary["training_loss"], label="Training")
    ax.plot(summary["validation_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fname = "plots/training_loss.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")

    # PIT calibration
    print("Computing PIT calibration...")
    n_test = 2000
    theta_test, x_test = simulate_dataset(n_test)
    pit_values = np.empty((n_test, len(PARAM_LABELS)))

    for i in trange(n_test, desc="PIT"):
        samples_i = npe.sample(x_test[i], n_samples=2000, show_progress_bars=False)
        pit_values[i] = np.mean(samples_i < theta_test[i], axis=0)

    fig, axes = plt.subplots(1, len(PARAM_LABELS), figsize=(12, 3))
    for j, (ax, label) in enumerate(zip(axes, PARAM_LABELS)):
        ax.hist(pit_values[:, j], bins=20, density=True, edgecolor="k", alpha=0.7)
        ax.axhline(1.0, color="k", ls="--")
        ax.set_xlabel(f"PIT({label})")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
    fname = "plots/pit_calibration.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
