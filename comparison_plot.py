"""Single-parameter comparison: NPE, MCMC, and true value for Mock B."""

import glob, numpy as np, matplotlib.pyplot as plt
from example_transit import make_mock, npe_sample, run_numpyro_mcmc, load_npe, TRUE_MOCK_B
from transit_sbi import PARAM_LABELS

idx = PARAM_LABELS.index("rp_rs")
truth = TRUE_MOCK_B[idx]

npe, _ = load_npe(sorted(glob.glob("weights/npe_robust_*.pkl"))[-1])
x_obs, t_grid, _ = make_mock(TRUE_MOCK_B, seed=1)

npe_rp = npe_sample(npe, x_obs, n_samples=10_000)[:, idx]
mcmc_rp = run_numpyro_mcmc("Mock B", x_obs, t_grid, init_theta=TRUE_MOCK_B, seed_offset=1)[0][:, idx] # using mock B's rp/rs parameter for comparison

bins = np.linspace(min(npe_rp.min(), mcmc_rp.min()), max(npe_rp.max(), mcmc_rp.max()), 60)
c = 0.5 * (bins[1:] + bins[:-1])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(c, np.histogram(npe_rp, bins=bins, density=True)[0], "C0", lw=2, label="NPE")
ax.plot(c, np.histogram(mcmc_rp, bins=bins, density=True)[0], "C2", lw=2, label="MCMC")
ax.axvline(truth, color="red", ls="--", lw=2, label=f"True = {truth}")
ax.set(xlabel=r"$R_p/R_\star$", ylabel="Posterior density",
       title=r"Posterior benchmarking for $R_p/R_\star$ (Mock B)")
ax.legend()

fig.savefig("plots/comparison_rp_rs_mock_B.png", dpi=150, bbox_inches="tight")
print("Saved plots/comparison_rp_rs_mock_B.png")
