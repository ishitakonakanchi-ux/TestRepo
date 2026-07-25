"""NPE vs MCMC vs Truth for Mock B, rp_rs parameter."""
import glob, numpy as np, matplotlib.pyplot as plt
from npe_wrapper import NPEEstimator
from transit_sbi import simulator, N_OBS

IDX = 2  # rp_rs
TRUE_B = np.array([0.55, 0.25, 0.17, 0.50, 0.30, 0.0, np.log10(2e-3)])

mcmc_rp = np.load("data/mcmc_cache/mock_B.npz")["samples"][:, IDX]

flux_err = np.full(N_OBS, 5e-4)
sigma = np.sqrt(flux_err**2 + 10.0 ** (2.0 * TRUE_B[-1]))
np.random.seed(0)
x = np.array(simulator(TRUE_B)) + np.random.normal(0, sigma, N_OBS)

npe = NPEEstimator().load(sorted(glob.glob("weights/npe_robust_*.pkl"))[-1])
npe_rp = npe.sample(x, n_samples=10_000, show_progress_bars=False)[:, IDX]

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(min(npe_rp.min(), mcmc_rp.min()), max(npe_rp.max(), mcmc_rp.max()), 60)
c = 0.5 * (bins[1:] + bins[:-1])
ax.plot(c, np.histogram(npe_rp, bins=bins, density=True)[0], "C0", lw=2, label="NPE")
ax.plot(c, np.histogram(mcmc_rp, bins=bins, density=True)[0], "C2", lw=2, label="MCMC")
ax.axvline(0.17, color="red", ls="--", lw=2, label="True = 0.17")
ax.set(xlabel=r"$R_p/R_\star$", ylabel="Density",
       title=r"Posterior benchmarking for $R_p/R_\star$ (Mock B)")
ax.legend()
fig.savefig("plots/comparison_rp_rs_mock_B.png", dpi=150, bbox_inches="tight")
print("Saved plots/comparison_rp_rs_mock_B.png")
