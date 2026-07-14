# Transit Light Curve Parameter Inference with SBI

Simulation-based inference (SBI) for exoplanet transit parameters using Neural Posterior Estimation (NPE).

## Overview

This project demonstrates how to use neural density estimation to infer transit parameters from photometric light curves. Instead of running MCMC for each new observation, we train a neural network once on simulated data, then obtain posteriors instantly via a single forward pass.

### Parameters

The simulator samples seven transit parameters. Full models infer all seven;
the core models infer the four bold parameters and marginalize the remaining
three nuisance parameters through simulation:

| Parameter | Symbol | Prior | Description |
|-----------|--------|-------|-------------|
| **Impact parameter** | `b` | U(0, 0.99) | Projected distance at mid-transit |
| **Transit duration** | `duration` | U(0.01, 0.35) days | Total transit duration |
| **Radius ratio** | `rp_rs` | U(0.003, 0.40) | Planet-to-star radius ratio |
| Limb darkening | `q1` | U(0.001, 1) | First Kipping coefficient |
| Limb darkening | `q2` | U(0, 1) | Second Kipping coefficient |
| **Transit-time offset** | `t0` | U(-0.05, 0.05) days | Mid-transit offset on the local grid |
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
./run_pipeline.sh                 # build, train, refine, PIT (in order)
./run_pipeline.sh train apply     # skip the library build
./run_pipeline.sh refine          # guarded NPE-IS using newest robust weights
./run_pipeline.sh check           # lightweight executable checks
COMPRESSION=core ./run_pipeline.sh train apply pit
COMPRESSION=core_domain PIT_NOISE_MODEL=white ./run_pipeline.sh pit
WEIGHTS=weights/npe_robust_<timestamp>.pkl ./run_pipeline.sh refine
MAX_TARGETS=30 ./run_pipeline.sh  # smaller, quicker library build
./run_pipeline.sh --help          # full stage list and options
```

Stages are `build`, `train`, `apply`, `refine`, `pit`, and `check`. With no
arguments it runs `build train refine pit` in order, using the model's native
noise distribution for PIT. The refined stage already performs the
NPE-versus-NUTS comparison, so the unrefined `apply` stage is not also run by
default. Set `WEIGHTS` to pin an exact model for `apply`, `refine`, or `pit`;
otherwise the newest model compatible with `COMPRESSION` is selected. The
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
python train_sbi.py                        # robust embedded model (recommended)
python train_sbi.py --compression hybrid   # weighted + residuals
python train_sbi.py --compression full     # near-lossless conditioning
python train_sbi.py --compression embedded # learned full-context correction
python train_sbi.py --compression robust   # embedded + four-start fit
python train_sbi.py --compression core     # four targets; clean Gaussian noise
python train_sbi.py --compression core_domain # four targets; randomized noise
```

This trains a Neural Spline Flow on simulated transit light curves. The observed
Kepler fluxes are not training data. Each simulation geometrically mixes two
per-bin `flux_err` profiles from the DR25 library and adds independent noise
with `sigma_i^2 = flux_err_i^2 + jitter^2`. Jitter is sampled uniformly in
log10 space so low-noise examples are not overwhelmed by very large jitter.
Eighty per cent of the empirical profiles are used for training and 20% are
reserved for PIT diagnostics.

The `weighted` compressor contains the weighted approximate MLE, fitted
log10 jitter, and weighted Fisher widths (13 values). The `hybrid` compressor
appends the 50 whitened fit residuals (63 values). The `full` representation
also includes the fitted baseline and all 50 log flux errors, retaining the
information needed to reconstruct the input curve up to residual clipping.
The `embedded` representation uses the same inputs as `full`, but keeps the 14
fit/baseline values as a skip connection and learns a compact encoding of the
residuals and per-bin errors before the flow.
The default `robust` mode uses that same embedding but selects the best of four
duration/impact-parameter fit seeds, exposing separated transit-geometry modes
that a single local fit can miss.
The `core` mode uses the robust representation but predicts only impact
parameter, duration, radius ratio, and transit-time offset. Limb darkening and
jitter are still simulated, so their uncertainty is marginalized into those
four posteriors. `core_domain` additionally mixes clean curves with randomized
error rescaling, short-range correlated noise, baseline slopes, sparse
heavy-tailed outliers, and bins marked missing through large uncertainties.

Outputs saved to `weights/`:
- `npe_<mode>_<timestamp>.pkl` - trained posterior and compression metadata
- `plots/<mode>_training_loss.png` - training/validation loss curves

### 2. Run inference and diagnostics

```bash
python example_transit.py weights/npe_robust_<timestamp>.pkl
# or: python example_transit.py weights/npe_core_domain_<timestamp>.pkl

# Optional accuracy mode: guarded NPE importance sampling (full models only)
NPE_IMPORTANCE_REFINE=1 python example_transit.py \
    weights/npe_robust_<timestamp>.pkl

# Equivalent pipeline wrapper
WEIGHTS=weights/npe_robust_<timestamp>.pkl ./run_pipeline.sh refine
```

This loads the trained model and:
- Compares NPE posteriors for two synthetic observations
- Validates against MCMC (NumPyro NUTS)
- Compares NPE and MCMC on four held-out real Kepler targets by default
- Reports compression, NPE, and MCMC runtimes and the end-to-end speed-up

Corner plots are saved to `plots/<mode>_*.png`. Run `python pit_plot.py` for
the separate simulation-based PIT calibration diagnostic. Set
`PIT_NOISE_MODEL=white` or `PIT_NOISE_MODEL=domain` to cross-test a core model
outside its native training-noise regime. PIT statistics are written to CSV as
well as plotted.

The optional NPE-IS accuracy mode uses the amortized posterior as an importance
proposal and corrects it with the exact Gaussian transit likelihood. It draws
50,000 proposals in 10,000-sample batches, then uses Pareto-smoothed importance
sampling. A correction is used only when its effective sample size is at least
300, no individual sample carries more than 5% of the total weight, and the
Pareto tail diagnostic is at most 0.7; otherwise the raw NPE is kept and the
failure is printed. Outputs use the `plots/robust_is_*` prefix, so the fast
uncorrected plots are not overwritten. The proposal batch, minimum and maximum
budgets, ESS, weight, and Pareto thresholds can be changed with
`NPE_IMPORTANCE_BATCH`, `NPE_IMPORTANCE_MIN_PROPOSALS`,
`NPE_IMPORTANCE_MAX_PROPOSALS`, `NPE_IMPORTANCE_MIN_ESS`,
`NPE_IMPORTANCE_MAX_WEIGHT`, and `NPE_IMPORTANCE_MAX_PARETO_K`.

#### Current validation snapshot

On the three real targets whose reference NUTS chains pass R-hat, ESS, and
divergence checks, the clean `core` model removes the artificial high-impact
railing: its mean marginal KS for impact parameter is 0.071. Its duration and
timing posteriors remain over-dispersed, so the four-parameter mean KS is 0.168.
The `core_domain` model gives calibrated held-out PIT for impact parameter,
duration, and radius ratio under both white and randomized noise, although a
small timing bias remains.

The guarded NPE-IS mode is the accuracy-oriented result. Across all seven
marginals on the same three converged real targets, its mean KS is 0.029 and
its mean Wasserstein distance is 0.040 reference-posterior standard deviations.
In the latest end-to-end validation, refined inference took 6.7--10.0 seconds
per converged real target after compilation, versus 22--125 seconds for NUTS
(2--19x faster; machine load affects these timings). The unrefined amortized
NPE remains the sub-second option. For KIC 1026957 the correction is rejected
(Pareto k 0.757) and NUTS independently fails convergence; the high-impact raw
NPE result is retained and no catalogue value is treated as ground truth.

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
| `run_pipeline.sh` | One-command pipeline: `build` → `train` → `refine` → `pit` |
| `pit_plot.py` | PIT calibration diagnostic (amortized, batched) |
| `test_noise_aware.py` | Executable checks for all compression and noise modes |
| `next_steps.md` | Handoff notes for the DR25 DV data and SBI next steps |

## Architecture

The recommended NPE uses:

- **Compression**: a four-start weighted transit fit plus fit residuals and all per-bin errors
- **Embedding**: a summary skip connection plus a learned 32-value residual/error context
- **Density estimator**: NSF with 10 transforms and 128 hidden features
- **Training**: refreshed, chunk-compressed pools of heteroscedastic simulations with a ReduceLROnPlateau scheduler
- **Optional accuracy layer**: adaptive, ESS-guarded importance correction of
  the full NPE with the vectorized transit likelihood

## Configuration

In `train_sbi.py`:
- `DEVICE = "cpu"` by default
- Set `NPE_DEVICE=cuda` to train the neural flow on an available CUDA GPU
- CUDA runs cap CPU helper threads at two by default; set
  `NPE_CPU_THREADS=1` to use only one (values above two are rejected)
- Pool size, refresh interval, simulations per epoch, epochs, and patience can
  be overridden with the `NPE_POOL_SIZE`, `NPE_REFRESH_EVERY`,
  `NPE_SIMS_PER_EPOCH`, `NPE_EPOCHS`, and `NPE_PATIENCE` environment variables
- `SEED = 42` for reproducibility

## Google Colab

Inference is fast on CPU, but training the robust/core models is substantially
faster with a Colab GPU (`NPE_DEVICE=cuda`).

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
weights = sorted(glob.glob("weights/npe_robust_*.pkl"))[-1]
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
weights = sorted(glob.glob("weights/npe_robust_*.pkl"))[-1]
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
weights = sorted(glob.glob("weights/npe_robust_*.pkl"))[-1]
!python example_transit.py {weights}
```
