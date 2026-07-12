"""PIT calibration check for the amortized MLE-compression NPE.

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
    plots/pit.png
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import trange

from npe_wrapper import NPEEstimator
from transit_sbi import simulate_compressed, PARAM_LABELS

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.size": 12})

# LaTeX panel labels per parameter.
LATEX = {
    "b": r"$b$",
    "duration": r"$\mathrm{duration}$",
    "rp_rs": r"$r_\mathrm{p}/r_\star$",
    "q1": r"$q_1$",
    "q2": r"$q_2$",
    "t0": r"$t_0$",
    "scatter": r"$\mathrm{scatter}$",
}

N_TEST = 2000    # test draws from the prior
N_POST = 1000    # posterior samples per draw
N_BINS = 20
CHUNK = 250      # summaries sampled per batched flow pass
SEED = 0


if __name__ == "__main__":
    np.random.seed(SEED)
    w = (sys.argv[1] if len(sys.argv) == 2
         else max(glob.glob("weights/npe_mle_*.pkl"), key=os.path.getmtime))
    print(f"Loading {w}")
    npe = NPEEstimator().load(w)

    theta, summ = simulate_compressed(N_TEST)
    theta = np.asarray(theta)

    # Batched: one vectorised flow pass per chunk of summaries, not per draw.
    pit = np.empty((N_TEST, len(PARAM_LABELS)))
    for a in trange(0, N_TEST, CHUNK, desc="PIT"):
        b = min(a + CHUNK, N_TEST)
        post = npe.sample_batch(summ[a:b], n_samples=N_POST)   # (b-a, N_POST, 7)
        pit[a:b] = (post < theta[a:b, None, :]).mean(axis=1)

    # Per-bin std of a uniform histogram (binomial fluctuation), density units.
    sig = np.sqrt((1 / N_BINS) * (1 - 1 / N_BINS) / N_TEST) * N_BINS

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, lab in zip(axes.ravel(), PARAM_LABELS):
        ax.axhspan(1 - 2 * sig, 1 + 2 * sig, color="0.90", zorder=0)
        ax.axhspan(1 - sig, 1 + sig, color="0.75", zorder=0)
        ax.axhline(1.0, color="0.4", lw=0.8, ls="--", zorder=1)
        ax.hist(pit[:, PARAM_LABELS.index(lab)], bins=N_BINS, range=(0, 1),
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

    # Empty 8th panel becomes a compact legend key.
    key = axes.ravel()[-1]
    key.axis("off")
    key.legend(handles=[
        Patch(facecolor="C0", alpha=0.75, label=r"$\mathrm{PIT}$"),
        Line2D([0], [0], color="0.4", ls="--", label=r"$\mathrm{uniform}$"),
        Patch(facecolor="0.75", label=r"$1\sigma$"),
        Patch(facecolor="0.90", label=r"$2\sigma$"),
    ], loc="center", frameon=False, handlelength=1.5)

    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/pit.png", dpi=200)
    print("Saved plots/pit.png")
