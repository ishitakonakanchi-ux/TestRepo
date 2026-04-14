"""
Train an NPE on transit light curves and save the model.

Usage:
    python train_sbi.py

Outputs (in weights/):
    - npe_<timestamp>.pkl: trained posterior
    - summary_<timestamp>.pkl: training/validation loss curves
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
from sbi.utils import BoxUniform
from npe_wrapper import NPEEstimator
from transit_sbi import simulate_dataset, PRIOR_LOW, PRIOR_HIGH, SIGMA

# Set to "cuda" for GPU training
DEVICE = "cpu"
SEED = 42


class CNNEmbedding(nn.Module):
    """1D CNN to compress light curves into a low-dimensional summary.

    A CNN works well here because the light curve has temporal structure:
    local filters naturally detect the transit shape, ingress slope, and depth.
    """
    def __init__(self, output_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    prior = BoxUniform(low=torch.tensor(PRIOR_LOW), high=torch.tensor(PRIOR_HIGH),
                        device=DEVICE)
    embedding_net = CNNEmbedding(output_dim=32)

    npe = NPEEstimator(
        model="maf",
        hidden_features=128,
        num_transforms=8,
        learning_rate=1e-3,
        batch_size=128,
        stop_after_epochs=50,
        validation_fraction=0.3,
        embedding_net=embedding_net,
        device=DEVICE,
    )
    npe.fit_online(
        simulate_fn=lambda n: simulate_dataset(n, noiseless=True),
        sigma=SIGMA,
        prior=prior,
        n_sims_per_epoch=10000,
        n_epochs=1000,
        patience=50,
    )

    # Save model and training summary
    os.makedirs("weights", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm")
    model_fname = f"weights/npe_{timestamp}.pkl"
    summary_fname = f"weights/summary_{timestamp}.pkl"

    npe.save(model_fname)
    with open(summary_fname, "wb") as f:
        pickle.dump(npe.summaries_, f)

    print(f"Saved {os.path.abspath(model_fname)}")
    print(f"Saved {os.path.abspath(summary_fname)}")
