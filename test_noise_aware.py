"""Small executable checks for noise-aware compression and NPE bookkeeping."""

from types import MethodType

import numpy as np

import npe_wrapper
from npe_wrapper import NPEEstimator
from transit_sbi import (N_OBS, enable_x64, score_compress, simulate_compressed,
                         simulate_dataset, simulator,
                         _domain_randomized_observation)


def main():
    np.random.seed(0)
    profiles = np.vstack([
        np.linspace(1e-5, 1e-3, N_OBS),
        np.linspace(1e-3, 1e-5, N_OBS),
    ])
    theta, flux, flux_err = simulate_dataset(
        8, flux_err_profiles=profiles
    )
    weighted = np.asarray(score_compress(flux, flux_err, mode="weighted"))
    hybrid = np.asarray(score_compress(flux, flux_err, mode="hybrid"))
    full = np.asarray(score_compress(flux, flux_err, mode="full"))
    embedded = np.asarray(score_compress(flux, flux_err, mode="embedded"))
    robust = np.asarray(score_compress(flux, flux_err, mode="robust"))
    core = np.asarray(score_compress(flux, flux_err, mode="core"))
    core_domain = np.asarray(
        score_compress(flux, flux_err, mode="core_domain")
    )

    assert theta.shape == (8, 7)
    assert np.all((-6.0 <= theta[:, -1]) & (theta[:, -1] <= -2.0))
    assert len(np.unique(flux_err, axis=0)) > len(profiles)
    assert flux_err.min() >= profiles.min() * (1.0 - 1e-12)
    assert flux_err.max() <= profiles.max() * (1.0 + 1e-12)
    assert weighted.shape == (8, 13)
    assert hybrid.shape == (8, 13 + N_OBS)
    assert full.shape == (8, 14 + 2 * N_OBS)
    assert embedded.shape == full.shape
    assert robust.shape == full.shape
    assert core.shape == full.shape
    assert core_domain.shape == full.shape
    assert (weighted.dtype == hybrid.dtype == full.dtype == embedded.dtype
            == robust.dtype == np.float32)
    assert np.isfinite(weighted).all() and np.isfinite(hybrid).all()
    assert np.isfinite(full).all()
    assert np.isfinite(embedded).all()
    assert np.isfinite(robust).all()
    assert np.isfinite(core).all() and np.isfinite(core_domain).all()
    assert np.allclose(weighted, hybrid[:, :13])
    assert np.allclose(weighted, full[:, :13])
    assert np.array_equal(full, embedded)
    assert np.all((hybrid[:, 13:] >= -8) & (hybrid[:, 13:] <= 8))
    assert np.all((full[:, 14:14 + N_OBS] >= -8)
                  & (full[:, 14:14 + N_OBS] <= 8))
    assert np.all((robust[:, 14:14 + N_OBS] >= -8)
                  & (robust[:, 14:14 + N_OBS] <= 8))
    assert np.array_equal(robust, core)
    assert np.array_equal(robust, core_domain)

    # Core modes keep the same context but return only the four scientifically
    # central targets. Domain augmentation remains finite, mixes clean and
    # perturbed curves, and exposes its corruption switches for validation.
    core_theta, core_summ = simulate_compressed(
        8, mode="core", flux_err_profiles=profiles
    )
    assert core_theta.shape == (8, 4)
    assert core_summ.shape == (8, 14 + 2 * N_OBS)
    n_domain = 512
    domain_theta = np.tile(theta, (n_domain // len(theta), 1))
    model_flux = np.tile(np.asarray(simulator(theta[0])), (n_domain, 1))
    input_err = np.tile(profiles, (n_domain // len(profiles), 1))
    domain_flux, domain_err, domain_meta = _domain_randomized_observation(
        model_flux, domain_theta, input_err
    )
    assert domain_theta.shape == (n_domain, 7)
    assert domain_flux.shape == domain_err.shape == (n_domain, N_OBS)
    assert np.isfinite(domain_flux).all() and np.isfinite(domain_err).all()
    assert np.all(domain_err > 0)
    assert domain_meta["augmented"].any()
    assert (~domain_meta["augmented"]).any()
    assert domain_meta["correlated"].any()
    assert domain_meta["slope"].any()
    assert domain_meta["outlier_count"].sum() > 0
    assert domain_meta["missing_count"].sum() > 0
    assert np.allclose(domain_meta["error_scale"][~domain_meta["augmented"]], 1)

    try:
        simulate_dataset(1, flux_err_profiles=profiles, noise_model="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid noise model was accepted")
    for invalid_n in (0, -1, 1.5, True):
        try:
            simulate_dataset(invalid_n, flux_err_profiles=profiles)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid simulation count {invalid_n!r} accepted")
    try:
        simulate_compressed(1, mode="invalid", flux_err_profiles=profiles)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid compression mode was accepted")

    # NUTS enables x64, but application-time compression must remain identical
    # to the float32 map used to generate the NPE training set.
    with enable_x64(True):
        weighted_x64_caller = np.asarray(
            score_compress(flux, flux_err, mode="weighted")
        )
        robust_x64_caller = np.asarray(
            score_compress(flux, flux_err, mode="robust")
        )
    assert np.array_equal(weighted, weighted_x64_caller)
    assert np.array_equal(robust, robust_x64_caller)

    # A single central-transit start used to miss this short grazing solution.
    truth = np.array([0.94, 0.075, 0.05, 0.5, 0.3, 0.0005, -5.2])
    small = np.array([0.2, 0.12, 0.01, 0.4, 0.3, 0.0, -5.5])
    # Reuse the already-compiled batch shape to keep this executable check
    # lightweight on memory-constrained login nodes.
    special_flux = np.vstack([
        np.tile(np.asarray(simulator(truth)), (4, 1)),
        np.tile(np.asarray(simulator(small)), (4, 1)),
    ])
    special_err = np.vstack([
        np.full((4, N_OBS), 3e-5),
        np.full((4, N_OBS), 1e-5),
    ])
    special_summary = np.asarray(score_compress(
        special_flux, special_err, mode="weighted"
    ))
    summary = special_summary[0]
    assert abs(summary[1] - truth[1]) < 0.015
    assert abs(summary[2] - truth[2]) < 0.02
    assert summary[6] < -5.5

    summary = special_summary[4]
    assert abs(summary[2] - small[2]) < 0.005

    # Ensemble training must retain every member's history, and distributing a
    # non-divisible draw count across members must still return exactly the
    # requested number of samples.
    estimator = NPEEstimator()

    def fake_fit(self, *args, **kwargs):
        member = len(self.posteriors_)
        self.posterior_ = member
        self.summaries_ = [{"member": member}]
        return self

    estimator.fit_online = MethodType(fake_fit, estimator)
    estimator.fit_online_ensemble(
        None, 0.0, None, 1, 1, n_ensemble=3
    )
    assert len(estimator.posteriors_) == 3
    assert [item["member"] for item in estimator.summaries_] == [0, 1, 2]

    original_sample = npe_wrapper._sample_on_cpu

    def fake_sample(post, count, *args, **kwargs):
        return np.full((count, 1), post, dtype=float)

    npe_wrapper._sample_on_cpu = fake_sample
    try:
        ensemble_samples = estimator.sample(
            np.zeros(1), n_samples=5, show_progress_bars=False
        )
    finally:
        npe_wrapper._sample_on_cpu = original_sample
    assert ensemble_samples.shape == (5, 1)
    print("noise-aware compression check passed")


if __name__ == "__main__":
    main()
