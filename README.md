# Transit Light Curve Parameter Inference with SBI

Simulation-based inference (SBI) for exoplanet transit parameters using Neural Posterior Estimation (NPE).

## Overview

This project demonstrates how to use neural density estimation to infer transit parameters from photometric light curves. Instead of running MCMC for each new observation, we train a neural network once on simulated data, then obtain posteriors instantly via a single forward pass.

### Parameters

The model infers three parameters from a transit light curve:

| Parameter | Symbol | Prior | Description |
|-----------|--------|-------|-------------|
| Impact parameter | `b` | U(0, 0.9) | Projected distance at mid-transit |
| Transit duration | `duration` | U(0.05, 0.35) days | Total transit duration |
| Radius ratio | `rp_rs` | U(0.03, 0.25) | Planet-to-star radius ratio |

Fixed parameters: orbital period (3 days), quadratic limb darkening [0.3, 0.2].

## Installation

```bash
pip install sbi jax jaxoplanet emcee corner torch numpy matplotlib
```

## Usage

### 1. Train the NPE

```bash
python train_sbi.py
```

This trains a Masked Autoregressive Flow (MAF) with a CNN embedding network on simulated transit light curves. Training uses fresh simulations each epoch with on-the-fly noise augmentation.

Outputs saved to `weights/`:
- `npe_<timestamp>.pkl` - trained posterior (includes full network)
- `summary_<timestamp>.pkl` - training/validation loss curves

### 2. Run inference and diagnostics

```bash
python example_transit.py weights/npe_2026-04-14_15h30m.pkl
```

This loads the trained model and:
- Compares NPE posteriors for two synthetic observations
- Validates against MCMC (emcee)
- Generates posterior predictive light curves
- Computes PIT calibration diagnostics

Plots saved to `plots/`.

## Files

| File | Description |
|------|-------------|
| `transit_sbi.py` | Transit simulator using jaxoplanet |
| `npe_wrapper.py` | Sklearn-like wrapper around sbi's NPE |
| `train_sbi.py` | Training script |
| `example_transit.py` | Inference and diagnostic plots |

## Architecture

The NPE uses:
- **Embedding network**: 1D CNN (3 conv layers) to compress the 50-point light curve into a 32-dimensional summary
- **Density estimator**: MAF with 8 transforms and 128 hidden units
- **Training**: Online training with 10k fresh simulations per epoch, ReduceLROnPlateau scheduler

## Configuration

In `train_sbi.py`:
- `DEVICE = "cpu"` or `"cuda"` for GPU acceleration
- `SEED = 42` for reproducibility

## Google Colab

**First**: Runtime → Change runtime type → GPU (T4)

A test notebook `colab_test.ipynb` is included - upload it to Colab to verify the setup works.

### Option 1: Public repo

```python
# Install dependencies (JAX with CUDA for GPU)
# JAX is pre-installed on Colab with CUDA support
!pip install -q sbi jaxoplanet emcee corner

# Clone and setup
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
!sed -i 's/DEVICE = "cpu"/DEVICE = "cuda"/' train_sbi.py

# Train
!python train_sbi.py

# Find the weights file and run inference
import glob
weights = sorted(glob.glob("weights/npe_*.pkl"))[-1]
!python example_transit.py {weights}
```

### Option 2: Private repo (via GitHub token)

1. Create a Personal Access Token at https://github.com/settings/tokens (select `repo` scope)
2. In Colab:

```python
# Install dependencies
# JAX is pre-installed on Colab with CUDA support
!pip install -q sbi jaxoplanet emcee corner

# Clone private repo
from getpass import getpass
import os
token = getpass("GitHub token: ")
os.environ["GH_TOKEN"] = token
!git clone https://$GH_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
!sed -i 's/DEVICE = "cpu"/DEVICE = "cuda"/' train_sbi.py

# Train
!python train_sbi.py

# Inference
import glob
weights = sorted(glob.glob("weights/npe_*.pkl"))[-1]
!python example_transit.py {weights}
```

### Option 3: Upload files manually

1. Upload `transit_sbi.py`, `npe_wrapper.py`, `train_sbi.py`, `example_transit.py` to Colab
2. Run:

```python
# Install dependencies
# JAX is pre-installed on Colab with CUDA support
!pip install -q sbi jaxoplanet emcee corner

# Enable GPU and train
!sed -i 's/DEVICE = "cpu"/DEVICE = "cuda"/' train_sbi.py
!python train_sbi.py

# Inference
import glob
weights = sorted(glob.glob("weights/npe_*.pkl"))[-1]
!python example_transit.py {weights}
```
