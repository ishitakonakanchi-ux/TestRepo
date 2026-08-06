"""Accuracy & precision benchmark for rp_rs with color-coded objects."""
import csv, numpy as np, matplotlib.pyplot as plt

PARAM = "rp_rs"

names, npe_mean, npe_std, mcmc_mean, mcmc_std = [], [], [], [], []
with open("plots/robust_comparison.csv") as f:
    for r in csv.DictReader(f):
        if r["parameter"] == PARAM and r["nuts_converged"] == "True":
            names.append(r["name"])
            npe_mean.append(float(r["npe_mean"]))
            npe_std.append(float(r["npe_std"]))
            mcmc_mean.append(float(r["mcmc_mean"]))
            mcmc_std.append(float(r["mcmc_std"]))

npe_mean, npe_std = np.array(npe_mean), np.array(npe_std)
mcmc_mean, mcmc_std = np.array(mcmc_mean), np.array(mcmc_std)
n_objects = len(names)
print(f"Plotting {n_objects} converged objects")

# Assign a unique color per object using tab10 colormap
colors = plt.cm.tab10(np.linspace(0, 1, n_objects))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Accuracy plot ---
ax = axes[0]
for i, name in enumerate(names):
    ax.scatter(mcmc_mean[i], npe_mean[i], s=110, alpha=0.85,
               color=colors[i], edgecolor="k", zorder=3, label=name)
lo, hi = min(mcmc_mean.min(), npe_mean.min()), max(mcmc_mean.max(), npe_mean.max())
pad = 0.08 * (hi - lo)
xr = np.array([lo - pad, hi + pad])
ax.plot(xr, xr, "k--", alpha=0.6, label="y = x")
m, b = np.polyfit(mcmc_mean, npe_mean, 1)
ax.plot(xr, m * xr + b, "r-", alpha=0.7,
        label=f"best fit: y = {m:.3f}x + {b:.4f}")
ax.set(xlabel=r"MCMC mean of $R_p/R_\star$",
       ylabel=r"NPE mean of $R_p/R_\star$",
       title="Posterior Mean Benchmark", xlim=xr, ylim=xr)
ax.grid(alpha=0.3)

# --- Precision plot ---
ax = axes[1]
for i, name in enumerate(names):
    ax.scatter(mcmc_std[i], npe_std[i], s=110, alpha=0.85,
               color=colors[i], edgecolor="k", zorder=3, label=name)
lo, hi = min(mcmc_std.min(), npe_std.min()), max(mcmc_std.max(), npe_std.max())
pad = 0.08 * (hi - lo)
xr = np.array([lo - pad, hi + pad])
ax.plot(xr, xr, "k--", alpha=0.6, label="y = x")
m, b = np.polyfit(mcmc_std, npe_std, 1)
ax.plot(xr, m * xr + b, "r-", alpha=0.7,
        label=f"best fit: y = {m:.3f}x + {b:.4f}")
ax.set(xlabel=r"MCMC std of $R_p/R_\star$",
       ylabel=r"NPE std of $R_p/R_\star$",
       title="Precision Benchmark", xlim=xr, ylim=xr)
ax.grid(alpha=0.3)

# Shared legend on the right side of the figure
handles = [plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colors[i], markeredgecolor="k",
                       markersize=10, label=names[i])
           for i in range(n_objects)]
fig.legend(handles=handles, loc="center right", fontsize=9,
           bbox_to_anchor=(1.0, 0.5), title="Objects")

plt.suptitle(r"NPE vs MCMC benchmarking for $R_p/R_\star$", fontsize=14)
plt.tight_layout(rect=[0, 0, 0.85, 0.96])
fig.savefig("plots/benchmark_rp_rs.png", dpi=150, bbox_inches="tight")
print("Saved plots/benchmark_rp_rs.png")
