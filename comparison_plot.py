"""Relative accuracy/precision benchmark for rp_rs with best-fit lines."""
import csv, numpy as np, matplotlib.pyplot as plt

PARAM = "rp_rs"
npe_mean, npe_std, mcmc_mean, mcmc_std = [], [], [], []
with open("plots/robust_comparison.csv") as f:
    for r in csv.DictReader(f):
        if r["parameter"] == PARAM and r["nuts_converged"] == "True":
            npe_mean.append(float(r["npe_mean"]))
            npe_std.append(float(r["npe_std"]))
            mcmc_mean.append(float(r["mcmc_mean"]))
            mcmc_std.append(float(r["mcmc_std"]))

npe_mean = np.array(npe_mean); npe_std = np.array(npe_std)
mcmc_mean = np.array(mcmc_mean); mcmc_std = np.array(mcmc_std)

def plot_with_fit(ax, x, y, color, xlabel, ylabel, title):
    # scatter
    ax.scatter(x, y, s=100, alpha=0.75, color=color, edgecolor="k", zorder=3)
    
    lo = min(x.min(), y.min()) #range
    hi = max(x.max(), y.max())
    pad = 0.08 * (hi - lo)
    xr = np.array([lo - pad, hi + pad])

    ax.plot(xr, xr, "k--", alpha=0.6, label="y = x (perfect agreement)")
    
    # best-fit line
    m, b = np.polyfit(x, y, 1)
    ax.plot(xr, m * xr + b, "r-", alpha=0.8,
            label=f"best fit: y = {m:.3f}x + {b:.4f}")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title,
           xlim=xr, ylim=xr)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

plot_with_fit(axes[0], mcmc_mean, npe_mean, "C0",
              r"MCMC mean of $R_p/R_\star$",
              r"NPE mean of $R_p/R_\star$",
              "Accuracy comparison")

plot_with_fit(axes[1], mcmc_std, npe_std, "C2",
              r"MCMC std of $R_p/R_\star$",
              r"NPE std of $R_p/R_\star$",
              "Precision comparison")

plt.suptitle(r"NPE vs MCMC benchmarking for $R_p/R_\star$", fontsize=14)
plt.tight_layout()
fig.savefig("plots/comparison_rp_rs.png", dpi=150, bbox_inches="tight")
print("Saved plots/comparison_rp_rs.png")
