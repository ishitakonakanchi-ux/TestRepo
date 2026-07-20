"""PIT calibration check for a noise-aware amortized transit NPE.

For many test draws theta* ~ prior, simulate a noisy curve, compress it, sample
the NPE posterior, and form the per-parameter PIT

    PIT_i = P(theta_i < theta*_i | x) = mean(posterior_i < theta*_i).

A calibrated posterior gives Uniform(0,1) PITs -> flat histogram. Shape diagnoses
miscalibration: U -> overconfident (too narrow), inverted-U -> too broad,
slope/shift -> biased. Amortized, so this is cheap: one forward pass per draw.

Sampling goes through npe.sample() (not npe.posterior_.sample) so the logit ->
box transform is applied, matching how the model is used on real data.

Usage:
    python pit_plot.py [weights_file]

Outputs:
    plots/pit_<compression>.png
    plots/pit_<compression>.csv
"""
import glob
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import kstest
from tqdm import trange

from npe_wrapper import NPEEstimator
from transit_sbi import (COMPRESSION_MODES, load_flux_err_profiles,
                         simulate_compressed, simulate_dataset, score_compress,
                         PARAM_LABELS, CORE_MODES)

plt.rcParams.update({"font.family": "serif", "font.size": 12})

# LaTeX panel labels per parameter.
LATEX = {
    "b": r"$b$",
    "duration": r"$\mathrm{duration}$",
    "rp_rs": r"$r_\mathrm{p}/r_\star$",
    "q1": r"$q_1$",
    "q2": r"$q_2$",
    "t0": r"$t_0$",
    "log10_jitter": r"$\log_{10}\,\mathrm{jitter}$",
}

N_TEST = 2000    # test draws from the prior
N_POST = 1000    # posterior samples per draw
N_BINS = 20
CHUNK = 250      # summaries sampled per batched flow pass
SEED = 0


if __name__ == "__main__":
    np.random.seed(SEED)
    if len(sys.argv) > 2:
        sys.exit("Usage: python pit_plot.py [weights_file]")
    requested = os.environ.get("COMPRESSION")
    if requested is not None and requested not in COMPRESSION_MODES:
        sys.exit(f"Unknown COMPRESSION mode: {requested}")
    modes = (requested,) if requested in COMPRESSION_MODES else COMPRESSION_MODES
    candidates = [path for mode in modes
                  for path in glob.glob(f"weights/npe_{mode}_20*.pkl")]
    if len(sys.argv) != 2 and not candidates:
        sys.exit("No noise-aware weights found. Run train_sbi.py first.")
    w = (sys.argv[1] if len(sys.argv) == 2
         else max(candidates, key=os.path.getmtime))
    print(f"Loading {w}")
    npe = NPEEstimator().load(w)
    compression = npe.metadata_.get("compression")
    expected_schema = 5 if compression in CORE_MODES else 4
    if (compression not in COMPRESSION_MODES
            or npe.metadata_.get("schema_version") != expected_schema):
        sys.exit("This model uses older preprocessing; retrain it.")
    target_labels = list(npe.metadata_.get("target_labels", PARAM_LABELS))
    target_indices = list(npe.metadata_.get(
        "target_indices", range(len(PARAM_LABELS))
    ))
    if (len(target_labels) != len(target_indices)
            or len(set(target_indices)) != len(target_indices)
            or any(i < 0 or i >= len(PARAM_LABELS) for i in target_indices)
            or target_labels != [PARAM_LABELS[i] for i in target_indices]):
        sys.exit("The model has invalid target-parameter metadata; retrain it.")

    profiles = load_flux_err_profiles()
    if len(profiles) != npe.metadata_.get("error_profile_count"):
        sys.exit("The Kepler library changed after training; retrain the model.")
    holdout = np.asarray(
        npe.metadata_.get("holdout_profile_indices", []), dtype=int
    )
    if len(holdout) == 0:
        sys.exit("The model has no held-out error profiles; retrain it.")
    if (len(np.unique(holdout)) != len(holdout)
            or np.any((holdout < 0) | (holdout >= len(profiles)))):
        sys.exit("The model has invalid held-out profile indices; retrain it.")
    print(f"Testing on {len(holdout)} held-out Kepler error profiles")
    requested_noise = os.environ.get("PIT_NOISE_MODEL", "native")
    native_noise = "domain" if compression == "core_domain" else "white"
    if requested_noise not in ("native", "white", "domain"):
        sys.exit("PIT_NOISE_MODEL must be native, white, or domain")
    test_noise = native_noise if requested_noise == "native" else requested_noise
    print(f"PIT noise model: {test_noise} (native: {native_noise})")
    if test_noise == native_noise:
        theta, summ = simulate_compressed(
            N_TEST, mode=compression, flux_err_profiles=profiles[holdout]
        )
    else:
        theta_full, flux, flux_err = simulate_dataset(
            N_TEST, flux_err_profiles=profiles[holdout],
            noise_model=test_noise,
        )
        theta = theta_full[:, target_indices]
        summ = np.asarray(score_compress(
            flux, flux_err, mode=compression
        ))
    theta = np.asarray(theta)

    # Batched: one vectorised flow pass per chunk of summaries, not per draw.
    pit = np.empty((N_TEST, len(target_labels)))
    for a in trange(0, N_TEST, CHUNK, desc="PIT"):
        b = min(a + CHUNK, N_TEST)
        post = npe.sample_batch(summ[a:b], n_samples=N_POST)
        pit[a:b] = (post < theta[a:b, None, :]).mean(axis=1)

    print("PIT uniformity (Kolmogorov-Smirnov):")
    pit_results = []
    for i, lab in enumerate(target_labels):
        stat, pvalue = kstest(pit[:, i], "uniform")
        pit_results.append({
            "parameter": lab,
            "pit_mean": float(pit[:, i].mean()),
            "ks_statistic": float(stat),
            "ks_pvalue": float(pvalue),
            "n_test": N_TEST,
            "n_posterior": N_POST,
            "test_noise_model": test_noise,
        })
        print(
            f"  {lab:12s} mean={pit[:, i].mean():.4f} "
            f"KS={stat:.4f} p={pvalue:.4g}"
        )

    # Per-bin std of a uniform histogram (binomial fluctuation), density units.
    sig = np.sqrt((1 / N_BINS) * (1 - 1 / N_BINS) / N_TEST) * N_BINS

    ncols = min(4, len(target_labels))
    nrows = int(np.ceil(len(target_labels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.25 * ncols, 3.0 * nrows),
        sharex=True, sharey=True, constrained_layout=True, squeeze=False,
    )
    flat_axes = axes.ravel()
    for i, (ax, lab) in enumerate(zip(flat_axes, target_labels)):
        ax.axhspan(1 - 2 * sig, 1 + 2 * sig, color="0.90", zorder=0)
        ax.axhspan(1 - sig, 1 + sig, color="0.75", zorder=0)
        ax.axhline(1.0, color="0.4", lw=0.8, ls="--", zorder=1)
        ax.hist(pit[:, i], bins=N_BINS, range=(0, 1),
                density=True, color="C0", alpha=0.75, edgecolor="C0",
                lw=0.6, zorder=2)
        ax.text(0.05, 0.92, LATEX[lab], transform=ax.transAxes, va="top",
                fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel(r"$\mathrm{PIT}$")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\mathrm{density}$")
    for ax in flat_axes[len(target_labels):]:
        ax.axis("off")

    fig.legend(handles=[
        Patch(facecolor="C0", alpha=0.75, label=r"$\mathrm{PIT}$"),
        Line2D([0], [0], color="0.4", ls="--", label=r"$\mathrm{uniform}$"),
        Patch(facecolor="0.75", label=r"$1\sigma$"),
        Patch(facecolor="0.90", label=r"$2\sigma$"),
    ], loc="upper center", ncol=4, frameon=False, handlelength=1.5)

    os.makedirs("plots", exist_ok=True)
    # Keep cross-domain checks distinct from a model whose native mode happens
    # to have the same name (for example core-on-domain vs core_domain-native).
    suffix = "" if requested_noise == "native" else f"_on_{test_noise}"
    output = f"plots/pit_{compression}{suffix}.png"
    fig.savefig(output, dpi=200)
    print(f"Saved {output}")
    csv_output = output.removesuffix(".png") + ".csv"
    with open(csv_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pit_results[0])
        writer.writeheader()
        writer.writerows(pit_results)
    print(f"Saved {csv_output}")
