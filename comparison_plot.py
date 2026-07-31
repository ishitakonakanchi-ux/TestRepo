"""Accuracy benchmark: NPE mean vs MCMC mean for rp_rs across all objects."""
import csv, numpy as np, matplotlib.pyplot as plt

PARAM = "rp_rs"

npe_mean, mcmc_mean = [], []
with open("plots/robust_comparison.csv") as f:
    for r in csv.DictReader(f):
        if r["parameter"] == PARAM and r["nuts_converged"] == "True":
            npe_mean.append(float(r["npe_mean"]))
            mcmc_mean.append(float(r["mcmc_mean"]))

npe_mean = np.array(npe_mean)
mcmc_mean = np.array(mcmc_mean)
print(f"Plotting {len(npe_mean)} converged objects")

fig, ax = plt.subplots(figsize=(8, 7))


ax.scatter(mcmc_mean, npe_mean, s=110, alpha=0.75, color="C0", edgecolor="k", zorder=3)

#range
lo = min(mcmc_mean.min(), npe_mean.min())
hi = max(mcmc_mean.max(), npe_mean.max())
pad = 0.08 * (hi - lo)
xr = np.array([lo - pad, hi + pad])

# y = x reference
ax.plot(xr, xr, "k--", alpha=0.6, label="y = x (perfect agreement)")

# best-fit
m, b = np.polyfit(mcmc_mean, npe_mean, 1)
ax.plot(xr, m * xr + b, "r-", alpha=0.8, label=f"best fit: y = {m:.3f}x + {b:.4f}")

ax.set(xlabel=r"MCMC mean of $R_p/R_\star$",
       ylabel=r"NPE mean of $R_p/R_\star$",
       title=r"Accuracy benchmarking: NPE vs MCMC for $R_p/R_\star$",
       xlim=xr, ylim=xr)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig("plots/comparison_accuracy_rp_rs.png", dpi=150, bbox_inches="tight")
print("Saved plots/comparison_accuracy_rp_rs.png")
