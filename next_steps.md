# Kepler DR25 DV light curves: next steps

The current data path uses the official Kepler DR25 Data Validation products.
This is deliberate.
For the first SBI application we should not clean raw or PDCSAP light curves ourselves, because that would make the project depend on a preprocessing pipeline rather than on the inference method.

## What the script downloads

The script `build_dr25_dv_library.py` queries the NASA Exoplanet Archive DR25 Threshold Crossing Event catalogue and downloads the corresponding Kepler Data Validation time-series files.
These files have names like

```text
kplr008359498-20160128150956_dvt.fits
```

Each file is an official Kepler DR25 `*_dvt.fits` product for one KIC target.
It is not a small transit cutout.
It contains the Data Validation time series over the Kepler baseline and one or more TCE extensions.
The useful columns for this project are:

- `TIME`: cadence time in BKJD, i.e. BJD minus 2454833.
- `PHASE`: time relative to the detected transit centre, in days.
- `LC_DETREND`: the official DV detrended light curve.
- `MODEL_INIT`: the official DR25 DV transit model on the same cadence grid.

The script uses `1 + LC_DETREND` as the observed relative flux and `1 + MODEL_INIT` as the reference DV model.
It does not download raw pixels and it does not run Lightkurve detrending.

## What the script writes

Run the default 20-object library with

```bash
/Users/rstiskalek/Projects/Teaching/venv_teach/bin/python build_dr25_dv_library.py
```

The main output directory is

```text
data/dr25_dv_library/
```

The important files are:

- `selected_tces.csv`: the accepted TCEs that enter the library.
- `manifest.csv`: every TCE the script tried, including rejected and failed cases.
- `dr25_dv_sbi_library.npz`: the stacked arrays for training or testing.
- `curves/*_sbi_grid.csv`: one 50-point binned curve per accepted TCE.

The plots are:

- `plots/dr25_dv_library_overview.png`: binned flux and binned DV model for each accepted object.
- `plots/dr25_dv_library_errors.png`: the binned empirical errors.

The FITS files are cached in

```text
data/dv/
```

It is safe to delete `data/` and rerun the script.
The next run will re-download the required FITS files and rebuild the library.

## What counts as a good curve

The script walks through the DR25 TCE catalogue in decreasing DR25 model SNR and keeps the first accepted objects.
The default target count is 20, controlled by `--max-targets`.

A TCE is rejected if it matches any of these conditions:

- the matched KOI row is labelled `FALSE POSITIVE`,
- the matched KOI row has the significant-secondary flag,
- the odd and even transits disagree strongly,
- the binned DV flux is inconsistent with the binned DR25 DV transit model.

The last two filters are important because some high-SNR TCEs are eclipsing-binary-like signals or long-period, poorly sampled events.
For example, `KIC 006187893 / TCE 01` and `KIC 006422170 / TCE 01` are rejected because their binned flux does not follow the official DV transit model closely enough.

The final terminal summary reports how much catalogue scanning was needed, for example:

```text
Selected 20 accepted TCEs
Tried 655 TCEs: 20 accepted, 635 rejected, 0 failed
Catalog rows available: 34032
```

## Binned data and errors

For each accepted TCE, the script cuts a window around transit and bins the official DV points to a 50-point grid.
The default half-window is at least `0.2 d`.
For wider transits the script expands the window using the support of the DR25 DV model.

The binned flux is the median of the DV detrended points in each bin.
The binned model is the median of the DR25 DV model in the same bin.
The empirical error is

```text
flux_err = 1.253 * robust_scatter / sqrt(n_eff)
```

Here `robust_scatter` is a MAD-based scatter estimate inside the bin, and `n_eff` is the number of distinct transit epochs contributing to that bin.
This is more conservative than dividing by the number of cadences because cadences inside a folded bin are not independent draws from the astrophysical noise process.

The NPZ file contains the arrays most useful for modelling:

- `phase_time`: shape `(N, 50)`, bin centres in days.
- `flux`: shape `(N, 50)`, binned DV detrended flux.
- `flux_err`: shape `(N, 50)`, empirical binned error.
- `model`: shape `(N, 50)`, binned DR25 DV model.
- `n_eff`: shape `(N, 50)`, distinct transit epochs per bin.
- `name`, `kepid`, `tce_plnt_num`: object identifiers.
- `dv_period_days`, `dv_epoch_bkjd`, `dv_duration_hours`, `dv_depth_ppm`: DR25 DV metadata.

Minimal loading example:

```python
import numpy as np

library = np.load("data/dr25_dv_library/dr25_dv_sbi_library.npz")
phase_time = library["phase_time"]
flux = library["flux"]
flux_err = library["flux_err"]
names = library["name"]
```

The arrays have a fixed length of 50, but the values of `phase_time` need not be identical for every object.
This matters for training.
If the network sees only `flux`, it cannot know whether the 50 values cover `[-0.2, 0.2] d` or a wider automatically chosen transit window.

## What you should do next

Start with the default 20-object library and inspect the plots.
Then build a larger sample, for example

```bash
/Users/rstiskalek/Projects/Teaching/venv_teach/bin/python build_dr25_dv_library.py --max-targets 100
```

After every rebuild, open `plots/dr25_dv_library_overview.png` and check that the accepted curves look like single transit-like events.
Also check `plots/dr25_dv_library_errors.png` and reject or investigate objects with very large errors or very small `n_eff`.
You do not need to keep every DR25 TCE.
You need a reproducible set of standard DR25 transit curves on which SBI and MCMC can be compared fairly.

Use the 20-object sample for debugging only.
Use a larger accepted sample for the first serious SBI run.
If the larger sample contains too many deep eclipsing-binary-like events, rebuild with a planet-like depth cut, for example

```bash
/Users/rstiskalek/Projects/Teaching/venv_teach/bin/python build_dr25_dv_library.py --max-targets 100 --max-depth-ppm 50000
```

The first SBI experiment should use the same kind of automatically chosen grid as the DR25 data.
The most direct version is to draw one real `phase_time` and `flux_err` vector from the accepted library for each simulation, generate a noiseless transit model on that specific `phase_time` grid, and add heteroscedastic Gaussian noise using the drawn `flux_err`.
This means the simulated data inherit the same window sizes and bin locations as the downloaded Kepler curves.
It also tests whether an amortised posterior estimator can learn the same approximate likelihood used by a conventional MCMC baseline.

The first MCMC comparison should use exactly the same 50 binned points and the same diagonal Gaussian likelihood.
This is the fair comparison because SBI and MCMC then see the same data vector, same errors, and same forward model.
The comparison should report posterior agreement, posterior predictive coverage, wall time per object, and the amortisation break-even point.

## Code changes needed for time and flux inputs

The current SBI training code assumes a fixed time grid.
In `transit_sbi.py`, the simulator evaluates every light curve on the global

```python
t_obs = jnp.linspace(-0.2, 0.2, N_OBS)
```

In `train_sbi.py`, the embedding network receives only the flux array.
This means the network treats the first, tenth, and fiftieth flux values as if they always correspond to the same times.
That assumption is not correct once the DR25 builder uses object-specific windows.

The simplest useful change is to make the data vector contain both time and flux on the automatically chosen 50-point grid.
This still keeps every observation fixed-dimensional, which `sbi` needs, but lets the network know where each flux value was measured.
The grid should not be hard-coded to `[-0.2, 0.2] d` during training unless you also force the downloaded DR25 curves onto that same grid.

Implement this in three steps.

First, change the simulator so it accepts a time grid:

```python
def simulator(params, t_obs):
    b, duration, rp_rs, period, t0, u1, u2 = params
    orbit = TransitOrbit(
        period=period,
        duration=duration,
        time_transit=t0,
        impact_param=b,
        radius_ratio=rp_rs,
    )
    return 1.0 + limb_dark_light_curve(orbit, [u1, u2])(t_obs)
```

Then write a DR25-aware simulation function.
The recommended first implementation should load `phase_time` and `flux_err` from `dr25_dv_sbi_library.npz`, randomly select one library row per simulation, evaluate the noiseless transit on that selected `phase_time`, and add noise using the selected `flux_err`.
This makes the training grid automatically match the grid distribution in the Kepler sample.
The returned observation can be either a flattened vector

```python
x = np.concatenate([phase_time_i, noisy_flux_i])
```

or a two-channel array

```python
x = np.stack([phase_time_i, noisy_flux_i], axis=0)
```

The two-channel version is cleaner if you keep a CNN embedding.

An alternative is to choose the grid from each simulated dip rather than sampling a real DR25 grid.
To do that, generate the noiseless model on a dense temporary grid, find where the model is below the out-of-transit baseline by the same fractional-depth threshold used in `build_dr25_dv_library.py`, pad the detected transit width, and place 50 bin centres across that window.
This reproduces the logic used for the Kepler data, but it is more code and it does not automatically reproduce the empirical distribution of Kepler window sizes.
For the first version, sample real DR25 `phase_time` grids from the library.

Finally, update the embedding network in `train_sbi.py`.
If you use the two-channel representation, change the first convolution from

```python
nn.Conv1d(1, 32, kernel_size=15, padding=7)
```

to

```python
nn.Conv1d(2, 32, kernel_size=15, padding=7)
```

and change the `forward` method so it does not add a fake channel dimension when `x` already has shape `(batch, 2, 50)`.
For example:

```python
def forward(self, x):
    if x.ndim == 2:
        x = x.unsqueeze(1)
    x = self.conv(x)
    x = x.flatten(start_dim=1)
    return self.fc(x)
```

If you want to include the errors as network input as well, use three channels:

```python
x = np.stack([phase_time_i, noisy_flux_i, flux_err_i], axis=0)
```

and set the first convolution to `nn.Conv1d(3, 32, ...)`.
This is probably the best first production version because the uncertainty changes from object to object.

These changes handle automatically chosen time grids with the same number of binned points.
If you want to use truly variable-length arrays, `sbi` still needs a fixed-dimensional batch.
You then have two practical choices: resample every curve to a common fixed grid, or pad all curves to a maximum length and pass a mask as an additional channel.
The fixed 50-bin DR25 library avoids this complication, so use the fixed-length representation first.

## Suggested project sequence

1. Build and inspect a 100-object accepted DR25 library.
2. Modify `transit_sbi.py` so the simulator can evaluate a transit on an input `phase_time` grid.
3. Modify `train_sbi.py` so the network receives `[phase_time, flux, flux_err]` rather than `flux` alone.
4. Train on simulations that draw automatically chosen `phase_time` grids and `flux_err` vectors from the DR25 library.
5. Hold out several real DR25 curves and run posterior predictive checks.
6. Run MCMC on the same held-out binned curves with the same diagonal Gaussian likelihood.
7. Compare SBI and MCMC posteriors, posterior predictive residuals, effective samples, and wall time.

After the basic comparison works, the next useful variants are:

- compare `[flux]`, `[phase_time, flux]`, and `[phase_time, flux, flux_err]` network inputs,
- restrict to a planet-like subset using `--max-depth-ppm`, for example `50000`,
- increase `--max-targets` and check whether the accepted sample remains visually clean,
- run posterior predictive checks on held-out DR25 curves.

## Caveats to state in the paper

The DR25 DV curves are already pipeline products.
The analysis is therefore an SBI application to standard Kepler Data Validation light curves, not to raw Kepler pixels.

The empirical `flux_err` values are useful but approximate.
They do not fully encode correlated stellar variability, pipeline detrending uncertainty, or the covariance induced by folding and binning.
For a first paper this is acceptable if the MCMC baseline uses the same binned data and the same approximate likelihood.

The DR25 DV model is used as a quality-control reference and as a diagnostic curve.
The scientific inference should still define its own forward model and priors rather than treating the DR25 model as the posterior truth.
