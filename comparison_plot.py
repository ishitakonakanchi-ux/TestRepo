"""Benchmark plots for rp_rs with color-coding (kepler only)."""
import csv, numpy as np, matplotlib.pyplot as plt

PARAM = "rp_rs"
names, npe_mean, npe_std, mcmc_mean, mcmc_std = [], [], [], [], []
with open("plots/robust_comparison.csv") as f:
    for r in csv.DictReader(f):
        if (r["parameter"] == PARAM
            and r["nuts_converged"] == "True"
            and not r["name"].startswith("Mock")):
            names.append(r["name"])
            npe_mean.append(float(r["npe_mean"]))
            npe_std.append(float(r["npe_std"]))
            mcmc_mean.append(float(r["mcmc_mean"]))
            mcmc_std.append(float(r["mcmc_std"]))

npe_mean, npe_std = np.array(npe_mean), np.array(npe_std)
mcmc_mean, mcmc_std = np.array(mcmc_mean), np.array(mcmc_std)
n_objects = len(names)
colors = plt.cm.tab10(np.linspace(0, 1, n_objects))

def fit_with_uncertainty(x, y):
    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    m, b = coeffs
    m_err, b_err = np.sqrt(np.diag(cov))
    return m, b, m_err, b_err

def plot_panel(ax, x, y, xlabel, ylabel, title):
    for i, name in enumerate(names):
        ax.scatter(x[i], y[i], s=110, alpha=0.85, color=colors[i],
                   edgecolor="k", zorder=3)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.15 * (hi - lo)
    xr = np.array([lo - pad, hi + pad])
    yx_line, = ax.plot(xr, xr, "k--", alpha=0.6, label="y = x")
    m, b, m_err, b_err = fit_with_uncertainty(x, y)
    sign = "+" if b >= 0 else "−"
    fit_line, = ax.plot(xr, m * xr + b, "r-", alpha=0.7,
        label=f"best fit: y = ({m:.3f} ± {m_err:.3f})x {sign} ({abs(b):.4f} ± {b_err:.4f})")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=xr, ylim=xr)
    ax.legend(handles=[yx_line, fit_line], fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plot_panel(axes[0], mcmc_mean, npe_mean,
           r"MCMC mean of $R_p/R_\star$",
           r"NPE mean of $R_p/R_\star$",
           "Posterior Mean Benchmark")
plot_panel(axes[1], mcmc_std, npe_std,
           r"MCMC std of $R_p/R_\star$",
           r"NPE std of $R_p/R_\star$",
           "Precision Benchmark")

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
