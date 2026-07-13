"""Small executable check for the two noise-aware compression modes."""

import numpy as np

from transit_sbi import N_OBS, score_compress, simulate_dataset, simulator


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

    assert theta.shape == (8, 7)
    assert np.all((-6.0 <= theta[:, -1]) & (theta[:, -1] <= -2.0))
    assert len(np.unique(flux_err, axis=0)) > len(profiles)
    assert flux_err.min() >= profiles.min() * (1.0 - 1e-12)
    assert flux_err.max() <= profiles.max() * (1.0 + 1e-12)
    assert weighted.shape == (8, 13)
    assert hybrid.shape == (8, 13 + N_OBS)
    assert np.isfinite(weighted).all() and np.isfinite(hybrid).all()
    assert np.allclose(weighted, hybrid[:, :13])
    assert np.all((hybrid[:, 13:] >= -8) & (hybrid[:, 13:] <= 8))

    # A single central-transit start used to miss this short grazing solution.
    truth = np.array([0.94, 0.075, 0.05, 0.5, 0.3, 0.0005, -5.2])
    summary = np.asarray(score_compress(
        np.asarray(simulator(truth)), np.full(N_OBS, 3e-5)
    ))
    assert abs(summary[1] - truth[1]) < 0.015
    assert abs(summary[2] - truth[2]) < 0.02
    assert summary[6] < -5.5

    small = np.array([0.2, 0.12, 0.01, 0.4, 0.3, 0.0, -5.5])
    summary = np.asarray(score_compress(
        np.asarray(simulator(small)), np.full(N_OBS, 1e-5)
    ))
    assert abs(summary[2] - small[2]) < 0.005
    print("noise-aware compression check passed")


if __name__ == "__main__":
    main()
