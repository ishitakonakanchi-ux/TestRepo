"""
Time NPE vs MCMC inference across the 20-object library.
Prints per-object times with a summary at end of run.
"""
import glob
import os
import pickle
import time
import numpy as np
from pathlib import Path

from example_transit import (
    NPEPosterior,
    load_reference_library,
    prior_bounds_from_transit_sbi,
    run_numpyro_mcmc,
)
from train_sbi import MLECompressor, DEVICE
from transit_sbi import PARAM_LABELS


N_NPE_SAMPLES = 10_000

# loading npe model
print("Loading trained NPE model...")
candidates = sorted(glob.glob("weights/npe_mle_*.pkl"))
if not candidates:
    candidates = sorted(glob.glob("weights/npe_*.pkl"))
model_fname = max(candidates, key=os.path.getmtime)
print(f"  Using: {model_fname}\n")

with open(model_fname, "rb") as f:
    payload = pickle.load(f)
posterior = payload["posterior"]
bounds = payload["bounds"]

npe = NPEPosterior(posterior, bounds)
compressor = MLECompressor(device=DEVICE)

# loading reference library
library = load_reference_library()
n_objects = len(library["name"])

print(f"\n{'='*70}")
print(f"TIMING COMPARISON: NPE vs MCMC across {n_objects} objects")
print(f"{'='*70}\n")

# looping
mcmc_times = []
npe_times = []
mcmc_ess_list = []
names = []

for i in range(n_objects):
    name = str(library["name"][i])
    x_obs = np.asarray(library["flux"][i])
    t_grid = np.asarray(library["phase_time"][i])
    flux_err = np.asarray(library["flux_err"][i])
    
    # init form dr25 pipeline values
    depth_ppm = float(library["dv_depth_ppm"][i])
    rp_rs_init = np.sqrt(max(depth_ppm, 1.0) * 1e-6)
    duration_days = float(library["dv_duration_hours"][i]) / 24.0
    init_theta = np.array([
        0.6,                                    # b
        max(0.05, min(0.35, duration_days)),    # duration
        max(0.03, min(0.25, rp_rs_init)),       # rp_rs
        0.5, 0.3,                               # q1, q2
        0.0,                                    # t0
        5e-4,                                   # scatter
    ])
    
    print(f"--- Object {i}: {name} ---")
    
    # mcmc timimg
    try:
        mcmc_samples, max_rhat, min_ess, divergences, mcmc_time = run_numpyro_mcmc(
            name, x_obs, t_grid,
            init_theta=init_theta, seed_offset=i,
        )
        print(f"  MCMC: {mcmc_time:.2f}s  (min ESS: {min_ess:.0f}, R-hat: {max_rhat:.3f})")
        mcmc_times.append(mcmc_time)
        mcmc_ess_list.append(min_ess)
    except Exception as e:
        print(f"  MCMC: FAILED — {type(e).__name__}: {e}")
        continue
    
    # npe timing
    try:
        t0 = time.perf_counter()
        summary = compressor.compress(x_obs)
        npe_samples = npe.sample(summary, n_samples=N_NPE_SAMPLES)
        npe_time = time.perf_counter() - t0
        
        print(f"  NPE:  {npe_time*1000:.1f}ms  ({N_NPE_SAMPLES:,} samples)")
        print(f"  Speedup: {mcmc_time/npe_time:.1f}x\n")
        npe_times.append(npe_time)
        names.append(name)
    except Exception as e:
        print(f"  NPE: FAILED — {type(e).__name__}: {e}\n")
        mcmc_times.pop()
        mcmc_ess_list.pop()
        continue

# summary 
mcmc_times = np.array(mcmc_times)
npe_times = np.array(npe_times)
mcmc_ess_arr = np.array(mcmc_ess_list)

print(f"\n{'='*70}")
print(f"SUMMARY  (successfully processed: {len(mcmc_times)} / {n_objects} objects)")
print(f"{'='*70}\n")

print(f"MCMC times per object:")
print(f"  Mean:   {mcmc_times.mean():.2f} s")
print(f"  Std:    {mcmc_times.std():.2f} s")
print(f"  Min:    {mcmc_times.min():.2f} s")
print(f"  Max:    {mcmc_times.max():.2f} s")
print(f"  Total:  {mcmc_times.sum():.1f} s  ({mcmc_times.sum()/60:.1f} min)")
print(f"  Median effective sample size: {np.median(mcmc_ess_arr):.0f}\n")

print(f"NPE times per object (compression + sampling):")
print(f"  Mean:   {npe_times.mean()*1000:.1f} ms")
print(f"  Std:    {npe_times.std()*1000:.1f} ms")
print(f"  Min:    {npe_times.min()*1000:.1f} ms")
print(f"  Max:    {npe_times.max()*1000:.1f} ms")
print(f"  Total:  {npe_times.sum():.2f} s\n")

print(f"NPE per posterior sample:")
print(f"  ~{npe_times.mean()/N_NPE_SAMPLES*1e6:.2f} us per sample")
print(f"  ({N_NPE_SAMPLES:,} samples drawn per object)\n")

print(f"Overall speedup: {mcmc_times.sum()/npe_times.sum():.1f}x\n")
print(f"To process {len(mcmc_times)} objects:")
print(f"  MCMC: {mcmc_times.sum()/60:.1f} minutes")
print(f"  NPE:  {npe_times.sum():.2f} seconds")
print(f"\n{'='*70}\n")
