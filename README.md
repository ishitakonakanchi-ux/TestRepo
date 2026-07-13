# Transit Light Curve Parameter Inference with SBI

Simulation-based inference (SBI) for exoplanet transit parameters using Neural Posterior Estimation (NPE).

## Overview

This project demonstrates how to use neural density estimation to infer transit parameters from photometric light curves. Instead of running MCMC for each new observation, we train a neural network once on simulated data, then obtain posteriors instantly via a single forward pass.

### Parameters

The current simulator samples seven transit parameters from a light curve:

| Parameter | Symbol | Prior | Description |
|-----------|--------|-------|-------------|
| Impact parameter | `b` | U(0, 0.99) | Projected distance at mid-transit |
| Transit duration | `duration` | U(0.01, 0.35) days | Total transit duration |
| Radius ratio | `rp_rs` | U(0.003, 0.40) | Planet-to-star radius ratio |
| Limb darkening | `q1` | U(0.001, 1) | First Kipping coefficient |
| Limb darkening | `q2` | U(0, 1) | Second Kipping coefficient |
| Transit-time offset | `t0` | U(-0.05, 0.05) days | Mid-transit offset on the local grid |
| Extra jitter | `log10_jitter` | U(-6, -2) | Log10 noise added in quadrature; -3 means 0.001 |

The period is fixed because it is not identified by one phase-folded transit.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## Usage

### Quick start: the whole pipeline

`run_pipeline.sh` chains the main stages end to end:

```bash
./run_pipeline.sh                 # build, train, apply (in order)
./run_pipeline.sh train apply     # skip the library build
./run_pipeline.sh train           # retrain only
./run_pipeline.sh pit             # PIT calibration plot (opt-in; needs weights)
COMPRESSION=hybrid ./run_pipeline.sh train  # optional residual-augmented model
MAX_TARGETS=30 ./run_pipeline.sh  # smaller, quicker library build
./run_pipeline.sh --help          # full stage list and options
```

Stages are `build`, `train`, `apply`, and `pit`. With no arguments it runs
`build train apply` in order; `pit` is opt-in and never runs by default. The
sections below document each stage individually.

### 0. Build good Kepler DR25 light curves

Use the official Kepler DR25 Data Validation time-series products, not local
PDCSAP preprocessing.
The builder downloads DR25 `*_dvt.fits` files, keeps only TCEs that pass the
current quality filters, bins each accepted curve to the 50-point SBI grid, and
stores empirical heteroscedastic errors.

```bash
.venv/bin/python build_dr25_dv_library.py --max-depth-ppm 50000
```

Main outputs:

- `data/dr25_dv_library/selected_tces.csv`
- `data/dr25_dv_library/manifest.csv`
- `data/dr25_dv_library/dr25_dv_sbi_library.npz`
- `data/dr25_dv_library/curves/*_sbi_grid.csv`
- `plots/dr25_dv_library_overview.png`
- `plots/dr25_dv_library_errors.png`

By default the script reproducibly samples the DR25 catalogue and keeps 50
accepted objects. This gives a much broader set of noise profiles than selecting
only the highest-SNR targets. The recommended pipeline also limits transit
depth to 50,000 ppm, requires detection SNR at least 7.1, and requires at least
three contributing transit epochs per bin.
It rejects TCEs without a matched KOI candidate, matched KOI false positives,
significant odd/even mismatches, and
folded curves whose binned DV flux is inconsistent with the DR25 DV transit
model.
Rejected objects remain in `manifest.csv`; only accepted curves enter the CSV,
NPZ, and plots.
Uncached FITS downloads show a per-file progress bar.

Useful options:

```bash
# Build more accepted curves
.venv/bin/python build_dr25_dv_library.py --max-targets 200

# Build from already-cached FITS files only
.venv/bin/python build_dr25_dv_library.py --no-download

# Restrict to shallower transits
.venv/bin/python build_dr25_dv_library.py --max-depth-ppm 20000
```

### 1. Train the NPE

```bash
python train_sbi.py                        # weighted summaries (recommended)
python train_sbi.py --compression hybrid   # optional weighted + residuals
```

This trains a Neural Spline Flow on simulated transit light curves. The observed
Kepler fluxes are not training data. Each simulation geometrically mixes two
per-bin `flux_err` profiles from the DR25 library and adds independent noise
with `sigma_i^2 = flux_err_i^2 + jitter^2`. Jitter is sampled uniformly in
log10 space so low-noise examples are not overwhelmed by very large jitter.
Eighty per cent of the empirical profiles are used for training and 20% are
reserved for PIT diagnostics.

The default `weighted` compressor contains the weighted approximate MLE, fitted
log10 jitter, and weighted Fisher widths (13 values). The optional `hybrid`
compressor appends the 50 whitened fit residuals (63 values).

Outputs saved to `weights/`:
- `npe_<mode>_<timestamp>.pkl` - trained posterior and compression metadata
- `plots/<mode>_training_loss.png` - training/validation loss curves

### 2. Run inference and diagnostics

```bash
python example_transit.py weights/npe_weighted_<timestamp>.pkl
```

This loads the trained model and:
- Compares NPE posteriors for two synthetic observations
- Validates against MCMC (NumPyro NUTS)
- Compares NPE and MCMC on one real Kepler target
- Reports compression, NPE, and MCMC runtimes and the end-to-end speed-up

Corner plots are saved to `plots/<mode>_*.png`. Run `python pit_plot.py` for
the separate simulation-based PIT calibration diagnostic.

### 3. Run NumPyro MCMC on a downloaded DR25 curve

```bash
.venv/bin/python run_dr25_mcmc.py --index 0
```

This runs NumPyro NUTS on one accepted, binned DR25 curve, prints the NumPyro
summary table, saves chain samples to `data/dr25_mcmc/`, and writes a corner
plot to `plots/`.
Use `--scatter-mode fixed` for one fixed scatter estimated from the curve's
`flux_err` values, or `--scatter-mode inferred` for one constant scatter
sampled by NumPyro.

## Files

| File | Description |
|------|-------------|
| `transit_sbi.py` | Transit simulator using jaxoplanet |
| `npe_wrapper.py` | Sklearn-like wrapper around sbi's NPE |
| `train_sbi.py` | Training script |
| `example_transit.py` | Inference and diagnostic plots |
| `run_dr25_mcmc.py` | NumPyro NUTS for one downloaded DR25 curve |
| `build_dr25_dv_library.py` | Build accepted, binned Kepler DR25 DV light curves |
| `run_pipeline.sh` | One-command pipeline: `build` → `train` → `apply` (+ opt-in `pit`) |
| `pit_plot.py` | PIT calibration diagnostic (amortized, batched) |
| `test_noise_aware.py` | Executable check for both compression modes |
| `next_steps.md` | Handoff notes for the DR25 DV data and SBI next steps |

## Architecture

The NPE uses:
- **Compression**: weighted 13-value summary, optionally plus 50 whitened residuals
- **Density estimator**: NSF with 10 transforms and 128 hidden features
- **Training**: resampled pools of heteroscedastic simulations, ReduceLROnPlateau scheduler

## Configuration

In `train_sbi.py`:
- `DEVICE = "cpu"` by default
- `SEED = 42` for reproducibility

## Google Colab

The standard CPU runtime is sufficient; `train_sbi.py` defaults to CPU.

A test notebook `colab_test.ipynb` is included - upload it to Colab to verify the setup works.

### Option 1: Public repo

```python
# Install dependencies
!pip install -q sbi jaxoplanet numpyro corner

# Clone repo
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO

# Train
!python train_sbi.py

# Find the weights file and run inference
import glob
weights = sorted(glob.glob("weights/npe_weighted_*.pkl"))[-1]
!python example_transit.py {weights}
```

### Option 2: Private repo (via GitHub token)

1. Create a Personal Access Token at https://github.com/settings/tokens (select `repo` scope)
2. In Colab:

```python
# Install dependencies
!pip install -q sbi jaxoplanet numpyro corner

# Clone private repo
from getpass import getpass
import os
token = getpass("GitHub token: ")
os.environ["GH_TOKEN"] = token
!git clone https://$GH_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO

# Train
!python train_sbi.py

# Inference
import glob
weights = sorted(glob.glob("weights/npe_weighted_*.pkl"))[-1]
!python example_transit.py {weights}
```

### Option 3: Upload files manually

1. Upload `transit_sbi.py`, `npe_wrapper.py`, `train_sbi.py`, `example_transit.py` to Colab
2. Run:

```python
# Install dependencies
!pip install -q sbi jaxoplanet numpyro corner

# Train
!python train_sbi.py

# Inference
import glob
weights = sorted(glob.glob("weights/npe_weighted_*.pkl"))[-1]
!python example_transit.py {weights}
```
