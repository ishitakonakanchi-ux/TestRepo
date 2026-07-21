"""
Time NPE vs MCMC inference across the 20-object library.
Prints per-object time summary at end of run.
"""
import time
import numpy as np
from pathlib import Path


LIBRARY_PATH = "data/dr25_dv_library/dr25_dv_sbi_library.npz"
N_OBJECTS = 20
N_NPE_SAMPLES = 10_000

library = np.load(LIBRARY_PATH, allow_pickle=True)

print(f"\n{'='*70}")
print(f"TIMING COMPARISON: NPE vs MCMC across {N_OBJECTS} objects")
print(f"{'='*70}\n")


mcmc_times = []
npe_times = []
mcmc_ess_list = []
names = []

# load npe at start
print("Loading trained NPE model...")
import pickle, glob, os
candidates = glob.glob("weights/npe_mle_*.pkl")
if not candidates:
    candidates = glob.glob("weights/npe_*.pkl")
model_fname = max(candidates, key=os.path.getmtime)
print(f"  Using: {model_fname}\n")

with open(model_fname, "rb") as f:
    npe_bundle = pickle.load(f)
npe = npe_bundle["npe"] if isinstance(npe_bundle, dict) else npe_bundle

# importing mcmc runner
from example_transit import run_numpyro_mcmc


for i in range(N_OBJECTS):
    name = str(library["name"][i])
    x_obs = np.array(library["flux"][i])
    t_grid = np.array(library["phase_time"][i])
    
    # init from dr25 pipeline values
    init_kep = np.array([
        0.6,                                                # b
        float(library["dv_duration_hours"][i]) / 24.0,      # duration
        np.sqrt(float(library["dv_depth_ppm"][i]) * 1e-6),  # rp_rs
        0.5, 0.3,                                           # q1, q2
        0.0,                                                # t0
        5e-4,                                               # scatter
    ])
    
    print(f"--- Object {i}: {name} ---")
    
    # timing mcmc
    t0 = time.time()
    mcmc_samples = run_numpyro_mcmc(
        name, x_obs, t_grid,
        init_theta=init_kep, seed_offset=i,
    )
    mcmc_time = time.time() - t0
    
    # numpyro effective sample size
    mcmc_ess = mcmc_samples.shape[0] if hasattr(mcmc_samples, 'shape') else 4000
    print(f"  MCMC: {mcmc_time:.2f}s  (samples: {mcmc_ess})")
    
    # timing npe
    t0 = time.time()
    npe_samples = npe.sample(x_obs, n_samples=N_NPE_SAMPLES)
    npe_time = time.time() - t0
    
    print(f"  NPE:  {npe_time*1000:.1f}ms  ({N_NPE_SAMPLES:,} samples)")
    print(f"  Speedup: {mcmc_time/npe_time:.1f}×\n")
    
    mcmc_times.append(mcmc_time)
    npe_times.append(npe_time)
    mcmc_ess_list.append(mcmc_ess)
    names.append(name)

# summary (including std)
mcmc_times = np.array(mcmc_times)
npe_times = np.array(npe_times)

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}\n")

print(f"MCMC times per object:")
print(f"  Mean:   {mcmc_times.mean():.2f} s")
print(f"  Std:    {mcmc_times.std():.2f} s")
print(f"  Min:    {mcmc_times.min():.2f} s")
print(f"  Max:    {mcmc_times.max():.2f} s")
print(f"  Total:  {mcmc_times.sum():.1f} s  ({mcmc_times.sum()/60:.1f} min)\n")

print(f"NPE times per object:")
print(f"  Mean:   {npe_times.mean()*1000:.1f} ms")
print(f"  Std:    {npe_times.std()*1000:.1f} ms")
print(f"  Min:    {npe_times.min()*1000:.1f} ms")
print(f"  Max:    {npe_times.max()*1000:.1f} ms")
print(f"  Total:  {npe_times.sum():.2f} s\n")

print(f"NPE per sample:")
print(f"  ~{npe_times.mean()/N_NPE_SAMPLES*1e6:.2f} μs per sample")
print(f"  ({N_NPE_SAMPLES:,} samples per object)\n")

print(f"Overall speedup: {mcmc_times.sum()/npe_times.sum():.1f}×\n")
print(f"To process {N_OBJECTS} objects:")
print(f"  MCMC: {mcmc_times.sum()/60:.1f} minutes")
print(f"  NPE:  {npe_times.sum():.2f} seconds")
print(f"\n{'='*70}")
