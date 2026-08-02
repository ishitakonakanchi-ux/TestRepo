"""Timing comparison: MCMC time/ESS vs NPE time/sample for each object."""
import csv, numpy as np, matplotlib.pyplot as plt

N_NPE_SAMPLES = 10_000

seen = set()
names, mcmc_per_ess, npe_per_sample = [], [], []
with open("plots/robust_comparison.csv") as f:
    for r in csv.DictReader(f):
        slug = r["slug"]
        if slug in seen:
            continue
        if r["nuts_converged"] != "True":
            continue
        seen.add(slug)
        names.append(r["name"].replace("KIC ", ""))
        mcmc_time = float(r["mcmc_seconds"])
        mcmc_ess = float(r["mcmc_min_ess"])
        npe_time = float(r["compression_seconds"]) + float(r["npe_seconds"])
        mcmc_per_ess.append(mcmc_time / mcmc_ess)
        npe_per_sample.append(npe_time / N_NPE_SAMPLES)

mcmc_per_ess = np.array(mcmc_per_ess)
npe_per_sample = np.array(npe_per_sample)
print(f"Plotting {len(names)} objects")

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(names))
width = 0.38

ax.bar(x - width/2, mcmc_per_ess, width, label="MCMC (time / ESS)", color="C2", alpha=0.85)
ax.bar(x + width/2, npe_per_sample, width, label="NPE (time / n_samples)", color="C0", alpha=0.85)

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Time per independent sample (seconds, log scale)")
ax.set_title("NPE vs MCMC: cost per independent posterior sample")
ax.legend(fontsize=11, loc="upper right")
ax.grid(alpha=0.3, axis="y", which="both")

plt.tight_layout()
fig.savefig("plots/timing_bar_plot.png", dpi=150, bbox_inches="tight")
print("Saved plots/timing_bar_plot.png")
