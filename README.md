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
pip install sbi jax jaxoplanet emcee corner torch numpy matplotlib astropy
```

## Usage

### 0. Build good Kepler DR25 light curves

Use the official Kepler DR25 Data Validation time-series products, not local
PDCSAP preprocessing.
The builder downloads DR25 `*_dvt.fits` files, keeps only TCEs that pass the
current quality filters, bins each accepted curve to the 50-point SBI grid, and
stores empirical heteroscedastic errors.

```bash
/Users/rstiskalek/Projects/Teaching/venv_teach/bin/python build_dr25_dv_library.py
```

Main outputs:

- `data/dr25_dv_library/selected_tces.csv`
- `data/dr25_dv_library/manifest.csv`
- `data/dr25_dv_library/dr25_dv_sbi_library.npz`
- `data/dr25_dv_library/curves/*_sbi_grid.csv`
- `plots/dr25_dv_library_overview.png`
- `plots/dr25_dv_library_errors.png`

By default the script walks through the DR25 TCE catalogue by decreasing model
SNR and keeps the first 20 accepted objects.
It rejects matched KOI false positives, significant odd/even mismatches, and
folded curves whose binned DV flux is inconsistent with the DR25 DV transit
model.
Rejected objects remain in `manifest.csv`; only accepted curves enter the CSV,
NPZ, and plots.
Uncached FITS downloads show a per-file progress bar.

Useful options:

```bash
# Build more accepted curves
/Users/rstiskalek/Projects/Teaching/venv_teach/bin/python build_dr25_dv_library.py --max-targets 100

# Build from already-cached FITS files only
/Users/rstiskalek/Projects/Teaching/venv_teach/bin/python build_dr25_dv_library.py --no-download

# Try a planet-like subset
/Users/rstiskalek/Projects/Teaching/venv_teach/bin/python build_dr25_dv_library.py --max-depth-ppm 50000
```

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
| `build_dr25_dv_library.py` | Build accepted, binned Kepler DR25 DV light curves |
| `next_steps.md` | Handoff notes for the DR25 DV data and SBI next steps |

## Architecture

The NPE uses:
- **Embedding network**: 1D CNN (3 conv layers) to compress the 50-point light curve into a 32-dimensional summary
- **Density estimator**: MAF with 8 transforms and 128 hidden units
- **Training**: Online training with 10k fresh simulations per epoch, ReduceLROnPlateau scheduler

## Configuration

In `train_sbi.py`:
- `DEVICE` auto-detects: CUDA → MPS → CPU
- `SEED = 42` for reproducibility

## Google Colab

**First**: Runtime → Change runtime type → GPU (T4)

A test notebook `colab_test.ipynb` is included - upload it to Colab to verify the setup works.

### Option 1: Public repo

```python
# Install dependencies (JAX with CUDA for GPU)
# JAX is pre-installed on Colab with CUDA support
!pip install -q sbi jaxoplanet emcee corner

# Clone repo (device auto-detected)
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO

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

# Train (device auto-detected)
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

# Train (device auto-detected)
!python train_sbi.py

# Inference
import glob
weights = sorted(glob.glob("weights/npe_*.pkl"))[-1]
!python example_transit.py {weights}
```
