# SBI for exoplanet transit light curves: literature review

Applying NPE directly to transit photometric light curves (Rp/Rs, impact
parameter, duration) is largely unexplored — most SBI + exoplanet work focuses
on atmospheric retrieval from spectra. Below are the most relevant papers,
grouped by topic.


## Foundational references

These papers define the methods used in this project and are essential
background for readers unfamiliar with simulation-based inference or
normalizing flows.

- **Papamakarios et al. (2021)** — Comprehensive review of normalizing flows
  for density estimation and inference. Covers autoregressive flows (MAF),
  coupling layers, and continuous flows. The MAF architecture used in our
  pipeline is described in detail here.
  [arXiv:1912.02762](https://arxiv.org/abs/1912.02762)

- **Cranmer, Brehmer & Louppe (2020)** — Review of simulation-based inference
  (SBI), surveying neural posterior estimation (NPE), neural likelihood
  estimation, and neural ratio estimation. Provides the conceptual framework
  for why SBI is needed when the likelihood is intractable or expensive.
  [arXiv:1911.01429](https://arxiv.org/abs/1911.01429)

- **Greenberg, Nonnenmacher & Macke (2019)** — Introduces Automatic Posterior
  Transformation (APT / SNPE-C), the sequential NPE algorithm that uses
  atomic proposals to correct for the prior–proposal mismatch in multi-round
  inference. This is the algorithm behind the `sbi` package's sequential mode
  and is the principled way to sharpen posteriors without brute-forcing the
  training set size.
  [arXiv:1905.07488](https://arxiv.org/abs/1905.07488)

- **Luger et al. (2019)** — Introduces `starry`, which computes analytic
  occultation light curves using Green's theorem on spherical harmonic surface
  maps. This is the mathematical foundation underlying `jaxoplanet`, the
  differentiable transit simulator used in our pipeline.
  [arXiv:1810.06559](https://arxiv.org/abs/1810.06559)


## Microlensing light curves (closest structural analogue)

These are the best templates for transit NPE because microlensing and transit
light curves share the same inference topology: a parameterised dip/bump in a
time series.

- **Zhang, Bloom & Gaudi et al. (2020)** — Amortized NPE with a ResNet-GRU
  embedding on binary microlensing light curves. Trained on 10^6 noiseless
  simulations with on-the-fly noise augmentation. Uses a 20-block MAF with a
  mixture-of-8-Gaussians base distribution to handle multimodal posteriors
  (close-wide degeneracy). The embedding is a deep 1D ResNet (18 layers) +
  2-layer GRU that compresses ~14,400-point light curves. Matches MCMC and
  achieves 10^5 samples/second on GPU.
  [arXiv:2010.04156](https://arxiv.org/abs/2010.04156)

- **Zhang et al. (2021)** — Follow-up applying real-time NPE to Roman binary
  microlensing. Same architecture, demonstrated on more realistic scenarios.
  [arXiv:2102.05673](https://arxiv.org/abs/2102.05673)

- **Smyth, Perreault-Levasseur & Hezaveh (2025)** — Replaces the CNN with a
  transformer embedding for free-floating planet microlensing. The transformer
  handles irregular sampling more naturally, but for regular cadence (Kepler,
  TESS) a CNN is sufficient.
  [arXiv:2512.11687](https://arxiv.org/abs/2512.11687)


## Eclipsing binaries (same physics as transits)

- **Blaum Hough, Bloom & Zhang (2026)** — SBI on detached eclipsing binary
  light curves (essentially the same forward model as transits). Recovers
  stellar radii, inclination, and limb darkening. Currently an AAS abstract
  only; no full paper yet.
  [ADS](https://ui.adsabs.harvard.edu/abs/2026AAS...24715407B)


## Atmospheric retrieval (methodological templates)

These papers don't fit transit shapes but establish best practices for NPE in
the exoplanet domain.

- **Vasist et al. (2023)** — NPE with MAF for atmospheric retrieval from
  transmission/emission spectra. Uses the `sbi` package. Matches nested
  sampling posteriors, orders of magnitude faster.
  [arXiv:2301.06575](https://arxiv.org/abs/2301.06575)

- **Gebhard et al. (2025)** — Flow matching posterior estimation (FMPE) for
  exoplanet atmospheres. A newer generative architecture than normalizing flows
  with better scalability. Published with the `fm4ar` software package.
  [arXiv:2410.21477](https://arxiv.org/abs/2410.21477)

- **Aubin et al. (2023)** — Winning entry of the Ariel 2023 ML challenge.
  Normalizing flows trained on the Ariel simulator to infer 6 atmospheric
  parameters from spectra. Demonstrates data augmentation and training
  strategies for simulator-based workflows.
  [arXiv:2309.09337](https://arxiv.org/abs/2309.09337)

- **Ardévol Martínez et al. (2024)** — FlopPITy: addresses the prior mismatch
  problem in SBI with a calibration-aware pipeline. Relevant because transit
  models have parameter correlations (e.g. Rp/Rs and impact parameter through
  the transit duration).
  [arXiv:2401.04168](https://arxiv.org/abs/2401.04168)

- **Lueber et al. (2025)** — FASTER: combines NPE with Bayesian model
  comparison (Bayes factors between atmospheric models) in milliseconds. The
  same approach could compare transit models (e.g. with/without starspots).
  [arXiv:2502.18045](https://arxiv.org/abs/2502.18045)

- **Yip et al. (2024)** — Systematic comparison of variational inference and
  normalizing flows for spectroscopic retrieval. Finds NFs more reliable for
  multimodal posteriors.
  [DOI:10.3847/1538-4357/ad063f](https://doi.org/10.3847/1538-4357/ad063f)


## Other related work

- **Haldemann et al. (2023)** — Conditional invertible neural networks (cINNs)
  for exoplanet interior structure from mass and radius (downstream of transit
  fitting). Matches MCMC posteriors.
  [arXiv:2202.00027](https://arxiv.org/abs/2202.00027)

- **Himes et al. (2022)** — NN surrogate for the radiative transfer forward
  model plugged into standard MCMC (emulator + MCMC, not full SBI). Useful as
  a baseline contrast.
  [arXiv:2003.02430](https://arxiv.org/abs/2003.02430)

- **Cobb et al. (2019)** — Ensemble of Bayesian NNs for atmospheric retrieval.
  One of the first papers showing NNs can match nested sampling for exoplanet
  inference.
  [arXiv:1905.10659](https://arxiv.org/abs/1905.10659)

- **Orsini et al. (2025)** — Flow matching for atmospheric retrieval, applied
  to the Ariel benchmark. Independent confirmation of Gebhard et al.'s
  findings.
  [DOI:10.1109/ACCESS.2025.3594751](https://doi.org/10.1109/ACCESS.2025.3594751)

- **Ikhsan et al. (2026)** — LSTMs (not SBI) for inferring planet parameters
  from transit timing variations.
  [DOI:10.3847/PSJ/ae3e86](https://doi.org/10.3847/PSJ/ae3e86)


## Key takeaways for this project

1. **No published paper applies NPE to Kepler/TESS transit shape fitting.**
   The closest work is on microlensing (Zhang+2021) and eclipsing binaries
   (Blaum Hough+2026).

2. **1D CNN embeddings are the standard** for compressing regularly-sampled
   light curves before the flow. Transformers help mainly for irregular
   cadence.

3. **SBI can converge to MCMC** given enough simulations (~10^5) and a good
   embedding (Zhang+2021 demonstrates this). Sequential NPE is the principled
   way to converge without brute-forcing the training set size.

4. **Flow matching** (Gebhard+2025) is a newer alternative to MAF/NSF that may
   scale better, worth exploring if normalizing flows plateau.


## Concrete improvements for our pipeline

Informed by Zhang et al. (2020) and the results of our embedding experiments
(MLP vs CNN, varying n_sims), the following changes would improve our NPE
transit pipeline. Ordered by expected impact per effort.

### 1. Reduce N_OBS (high impact, trivial)

We currently sample 200 points over [-0.2, 0.2] days. The transit shape is
smooth and 200 points massively oversamples it. Reducing to **50-80 points**
shrinks the embedding input, speeds up training, and removes redundancy the
flow wastes capacity on. The information content barely changes — the
constraining power comes from ingress/egress shape, not from having many
points on the flat baseline.

Alternatively, keep 200 points but narrow the window to [-0.1, 0.1] days to
focus on the transit itself. Currently transits with `duration` up to 0.5 days
extend beyond our [-0.2, 0.2] window anyway.

### 2. On-the-fly noise augmentation (high impact, easy)

Zhang et al. simulate 10^6 noiseless light curves and draw fresh noise
realizations on the fly during each training epoch. This effectively gives
unlimited training data from a fixed simulation budget. We currently bake the
noise into the training set once, so the network can memorise specific noise
patterns.

Implementation: store noiseless simulations, add `np.random.normal(0, SIGMA)`
freshly each time the dataloader serves a batch. This requires a custom
dataloader or a wrapper around `NumpyLoader`.

### 3. Scale up simulations to 10^5 (moderate impact, cheap)

Our JAX simulator is fast (~3s for 10k sims). Going to 10^5 sims costs ~30s
and, combined with noise augmentation, should be enough to close most of the
remaining gap to MCMC. Zhang et al. use 10^6 for a harder 7-parameter problem;
our 3-parameter problem likely saturates sooner.

### 4. Larger MAF with mixture base (moderate impact, easy)

Zhang et al. use 20 MAF blocks with a mixture-of-8-Gaussians base. We have
8 blocks with a standard Gaussian base. More transforms improve expressivity;
a mixture base helps capture non-Gaussian / multimodal structure in the `b`
posterior (impact parameter has a hard boundary at b=0 and a soft degeneracy
with duration).

### 5. Sequential NPE (high impact, more work)

Single-round amortized NPE trains on samples from the full prior — most
simulations produce light curves unlike the observation. Sequential NPE
(SNPE-C, supported by the `sbi` package) uses the previous round's posterior
as the proposal for the next round, concentrating simulations where they
matter. This is the principled way to match MCMC precision without brute-
forcing the training set. Zhang et al. chose not to use it (they wanted full
amortization for real-time inference on many events), but for a single-
observation comparison against MCMC it would be very effective.
