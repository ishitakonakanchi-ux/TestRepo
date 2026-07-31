"""Accuracy and precision benchmark: NPE vs MCMC for rp_rs across all objects."""
import csv, numpy as np, matplotlib.pyplot as plt

PARAM = "rp_rs"

# loading rows for rp_rs
npe_mean, npe_std, mcmc_mean, mcmc_std = [], [], [], []
with open("plots/robust_comparison.csv") as f:
    for r in csv.DictReader(f):
        if r["parameter"] == PARAM and r["nuts_converged"] == "True":
            npe_mean.append(float(r["npe_mean"]))
            npe_std.append(float(r["npe_std"]))
            mcmc_mean.append(float(r["mcmc_mean"]))
            mcmc_std.append(float(r["mcmc_std"]))

npe_mean, npe_std = np.array(npe_mean), np.array(npe_std)
mcmc_mean, mcmc_std = np.array(mcmc_mean), np.array(mcmc_std)
print(f"Using {len(npe_mean)} converged objects")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# means/accuracy plot
ax = axes[0]
ax.scatter(mcmc_mean, npe_mean, s=90, alpha=0.75, color="C0", edgecolor="k", zorder=3)
lo, hi = min(mcmc_mean.min(), npe_mean.min()), max(mcmc_mean.max(), npe_mean.max())
pad = 0.08 * (hi - lo)
ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", alpha=0.6, label="y = x (perfect agreement)")
ax.set(xlabel=r"MCMC mean of $R_p/R_\star$",
       ylabel=r"NPE mean of $R_p/R_\star$",
       title="Accuracy comparison")
ax.legend()
ax.grid(alpha=0.3)

# std/precision plot
ax = axes[1]
ax.scatter(mcmc_std, npe_std, s=90, alpha=0.75, color="C2", edgecolor="k", zorder=3)
lo, hi = min(mcmc_std.min(), npe_std.min()), max(mcmc_std.max(), npe_std.max())
pad = 0.08 * (hi - lo)
ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", alpha=0.6, label="y = x (perfect agreement)")
ax.set(xlabel=r"MCMC std of $R_p/R_\star$",
       ylabel=r"NPE std of $R_p/R_\star$",
       title="Precision comparison")
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle(r"NPE vs MCMC benchmarking for $R_p/R_\star$", fontsize=14)
plt.tight_layout()
fig.savefig("plots/benchmark_rp_rs.png", dpi=150, bbox_inches="tight")
print("Saved plots/benchmark_rp_rs.png")
