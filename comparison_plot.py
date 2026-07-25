"""NPE vs MCMC vs Truth for Mock B's rp_rs parameter."""
import glob, pickle, numpy as np, matplotlib.pyplot as plt

IDX = 2  # rp_rs


mcmc_rp = np.load("data/mcmc_cache/mock_B.npz")["samples"][:, IDX]

#load npe and sample
with open(sorted(glob.glob("weights/npe_robust_*.pkl"))[-1], "rb") as f:
    posterior = pickle.load(f)["posterior"]


from transit_sbi import simulate_compressed
TRUE_MOCK_B = np.array([0.55, 0.25, 0.17, 0.5, 0.3, 0.0, -2.699])
_, summary = simulate_compressed(TRUE_MOCK_B[None, :], seed=1)

# sample NPE
import torch
npe_rp = posterior.sample((10_000,), x=torch.tensor(summary[0], dtype=torch.float32)).numpy()[:, IDX]


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
