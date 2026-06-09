#!/usr/bin/env python3
"""Run NumPyro MCMC on one binned Kepler DR25 DV transit curve."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
from jax import config

MPLCONFIGDIR = Path("data/matplotlib-cache")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
PYTHON_CACHE_DIR = Path("data/python-cache")
PYTHON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(PYTHON_CACHE_DIR.resolve()))
home_cache = Path.home() / "Library" / "Caches"
if not os.access(home_cache, os.W_OK):
    local_home = Path("data/python-home").resolve()
    (local_home / "Library" / "Caches").mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(local_home)
warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")

config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits import TransitOrbit
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value


DEFAULT_LIBRARY = Path("data/dr25_dv_library/dr25_dv_sbi_library.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NumPyro NUTS on one binned Kepler DR25 DV curve."
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=DEFAULT_LIBRARY,
        help="Path to dr25_dv_sbi_library.npz.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based row index in the DR25 library.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=1000,
        help="Number of NUTS warmup steps per chain.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of retained NUTS samples per chain.",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=2,
        help="Number of MCMC chains.",
    )
    parser.add_argument(
        "--chain-method",
        choices=("parallel", "sequential", "vectorized"),
        default="vectorized",
        help="How NumPyro runs multiple chains.",
    )
    parser.add_argument(
        "--target-accept-prob",
        type=float,
        default=0.9,
        help="Target NUTS acceptance probability.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=10,
        help="Maximum NUTS tree depth.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="JAX random seed.",
    )
    parser.add_argument(
        "--duration-prior-scale",
        type=float,
        default=None,
        help=(
            "Truncated-normal prior scale for duration in days. "
            "Default is max(0.02, 0.5 * DR25 duration)."
        ),
    )
    parser.add_argument(
        "--rp-prior-scale",
        type=float,
        default=0.08,
        help="Truncated-normal prior scale for Rp/Rstar.",
    )
    parser.add_argument(
        "--t0-prior-scale",
        type=float,
        default=0.03,
        help="Normal prior scale for transit-time offset in days.",
    )
    parser.add_argument(
        "--baseline-prior-scale",
        type=float,
        default=0.01,
        help="Normal prior scale for additive relative-flux baseline.",
    )
    parser.add_argument(
        "--jitter-scale-multiplier",
        type=float,
        default=10.0,
        help="Half-normal jitter scale in units of median flux_err.",
    )
    parser.add_argument(
        "--scatter-mode",
        choices=("per-bin", "fixed", "inferred"),
        default="per-bin",
        help=(
            "Use the per-bin flux_err vector, one fixed scatter inferred "
            "from flux_err, or one scatter parameter inferred by MCMC."
        ),
    )
    parser.add_argument(
        "--fixed-scatter-stat",
        choices=("median", "mean", "rms"),
        default="median",
        help=(
            "Statistic used by constant-scatter modes to infer one reference "
            "value from flux_err."
        ),
    )
    parser.add_argument(
        "--fixed-scatter-value",
        type=float,
        default=None,
        help="Override the fixed scatter value instead of inferring it from flux_err.",
    )
    parser.add_argument(
        "--scatter-prior-multiplier",
        type=float,
        default=10.0,
        help=(
            "For --scatter-mode inferred, use this multiplier times the "
            "flux_err-derived scatter as the half-normal prior scale."
        ),
    )
    parser.add_argument(
        "--fix-nuisance",
        action="store_true",
        help=(
            "Fix t0_days, baseline, jitter, and limb darkening. This leaves "
            "only b, duration_days, and rp_rs as sampled parameters."
        ),
    )
    parser.add_argument(
        "--fix-t0-days",
        type=float,
        nargs="?",
        const=0.0,
        default=None,
        help="Fix the local transit-time offset in days. Default value is 0.",
    )
    parser.add_argument(
        "--fix-baseline",
        type=float,
        nargs="?",
        const=0.0,
        default=None,
        help="Fix the additive relative-flux baseline. Default value is 0.",
    )
    parser.add_argument(
        "--fix-jitter",
        type=float,
        nargs="?",
        const=0.0,
        default=None,
        help="Fix the extra relative-flux jitter. Default value is 0.",
    )
    parser.add_argument(
        "--fix-limb-darkening",
        type=float,
        nargs=2,
        metavar=("U1", "U2"),
        default=None,
        help="Fix quadratic limb darkening to these u1 u2 values.",
    )
    parser.add_argument(
        "--fixed-u1",
        type=float,
        default=0.3,
        help="u1 used by --fix-nuisance when --fix-limb-darkening is omitted.",
    )
    parser.add_argument(
        "--fixed-u2",
        type=float,
        default=0.2,
        help="u2 used by --fix-nuisance when --fix-limb-darkening is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dr25_mcmc"),
        help="Directory for saved chain samples.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for the corner plot when --corner-plot is not set.",
    )
    parser.add_argument(
        "--corner-plot",
        type=Path,
        default=None,
        help="Path for the posterior corner plot.",
    )
    parser.add_argument(
        "--no-corner",
        action="store_true",
        help="Do not write a posterior corner plot.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save posterior samples to an NPZ file.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable NumPyro progress bars.",
    )
    parser.add_argument(
        "--diagonal-mass",
        action="store_true",
        help="Use diagonal instead of dense NUTS mass-matrix adaptation.",
    )
    return parser.parse_args()


def load_target(library_path: Path, index: int) -> dict:
    if not library_path.exists():
        raise FileNotFoundError(
            f"{library_path} does not exist. Build it with "
            "`python build_dr25_dv_library.py --window-mode fixed`."
        )

    library = np.load(library_path, allow_pickle=True)
    n_targets = len(library["name"])
    if index < 0 or index >= n_targets:
        raise IndexError(f"--index {index} is outside the library range 0..{n_targets - 1}")

    phase_time = np.asarray(library["phase_time"][index], dtype=float)
    flux = np.asarray(library["flux"][index], dtype=float)
    flux_err = np.asarray(library["flux_err"][index], dtype=float)
    finite = np.isfinite(phase_time) & np.isfinite(flux) & np.isfinite(flux_err)
    finite &= flux_err > 0
    if np.sum(finite) < 5:
        raise RuntimeError("Fewer than five finite binned points are available.")

    duration_hours = float(library["dv_duration_hours"][index])
    depth_ppm = float(library["dv_depth_ppm"][index])
    return {
        "index": index,
        "name": str(library["name"][index]),
        "label": str(library["label"][index]),
        "kepid": int(library["kepid"][index]),
        "tce_plnt_num": int(library["tce_plnt_num"][index]),
        "period_days": float(library["dv_period_days"][index]),
        "epoch_bkjd": float(library["dv_epoch_bkjd"][index]),
        "duration_days": duration_hours / 24.0,
        "duration_hours": duration_hours,
        "depth_ppm": depth_ppm,
        "n_transits": int(library["dv_n_transits"][index]),
        "window_days": float(library["window_days"][index]),
        "phase_time": phase_time[finite],
        "flux": flux[finite],
        "flux_err": flux_err[finite],
    }


def kipping_q_to_u(q1: jax.Array, q2: jax.Array) -> tuple[jax.Array, jax.Array]:
    sqrt_q1 = jnp.sqrt(q1)
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    return u1, u2


def fixed_value(args_value: float | None, default_value: float, fix_nuisance: bool):
    if args_value is not None:
        return args_value
    if fix_nuisance:
        return default_value
    return None


def fixed_limb_darkening(args: argparse.Namespace):
    if args.fix_limb_darkening is not None:
        return tuple(args.fix_limb_darkening)
    if args.fix_nuisance:
        return args.fixed_u1, args.fixed_u2
    return None


def make_model(args: argparse.Namespace, target: dict):
    period_days = target["period_days"]
    duration_ref = target["duration_days"]
    duration_scale = args.duration_prior_scale
    if duration_scale is None:
        duration_scale = max(0.02, 0.5 * duration_ref)

    rp_ref = float(np.sqrt(max(target["depth_ppm"], 1.0) * 1e-6))
    median_flux_err = float(np.nanmedian(target["flux_err"]))
    scatter_mode = target.get("scatter_mode", "per-bin")
    scatter_ref = float(target.get("scatter_value", median_flux_err))
    jitter_scale = max(args.jitter_scale_multiplier * scatter_ref, 1e-8)
    scatter_prior_scale = max(args.scatter_prior_multiplier * scatter_ref, 1e-8)
    fixed_t0_days = fixed_value(args.fix_t0_days, 0.0, args.fix_nuisance)
    fixed_baseline = fixed_value(args.fix_baseline, 0.0, args.fix_nuisance)
    fixed_jitter = fixed_value(args.fix_jitter, 0.0, args.fix_nuisance)
    if scatter_mode == "inferred":
        fixed_jitter = 0.0
    fixed_limb = fixed_limb_darkening(args)

    def model(phase_time, flux=None, flux_err=None):
        b = numpyro.sample("b", dist.Uniform(0.0, 1.2))
        duration_days = numpyro.sample(
            "duration_days",
            dist.TruncatedNormal(
                duration_ref,
                duration_scale,
                low=0.01,
                high=max(0.5, 2.5 * duration_ref),
            ),
        )
        rp_rs = numpyro.sample(
            "rp_rs",
            dist.TruncatedNormal(rp_ref, args.rp_prior_scale, low=0.005, high=0.5),
        )
        if fixed_t0_days is None:
            t0_days = numpyro.sample("t0_days", dist.Normal(0.0, args.t0_prior_scale))
        else:
            t0_days = jnp.asarray(fixed_t0_days)

        if fixed_limb is None:
            q1 = numpyro.sample("q1", dist.Uniform(0.0, 1.0))
            q2 = numpyro.sample("q2", dist.Uniform(0.0, 1.0))
            u1, u2 = kipping_q_to_u(q1, q2)
        else:
            u1, u2 = (jnp.asarray(fixed_limb[0]), jnp.asarray(fixed_limb[1]))

        if fixed_baseline is None:
            baseline = numpyro.sample(
                "baseline", dist.Normal(0.0, args.baseline_prior_scale)
            )
        else:
            baseline = jnp.asarray(fixed_baseline)

        if scatter_mode == "inferred":
            scatter = numpyro.sample("scatter", dist.HalfNormal(scatter_prior_scale))
            jitter = jnp.asarray(0.0)
            sigma = scatter
        elif fixed_jitter is None:
            jitter = numpyro.sample("jitter", dist.HalfNormal(jitter_scale))
            sigma = jnp.sqrt(flux_err**2 + jitter**2)
        else:
            jitter = jnp.asarray(fixed_jitter)
            sigma = jnp.sqrt(flux_err**2 + jitter**2)

        orbit = TransitOrbit(
            period=period_days,
            duration=duration_days,
            time_transit=t0_days,
            impact_param=b,
            radius_ratio=rp_rs,
        )
        transit_flux = 1.0 + limb_dark_light_curve(orbit, [u1, u2])(phase_time)

        numpyro.deterministic("u1", u1)
        numpyro.deterministic("u2", u2)
        numpyro.deterministic("t0_days_value", t0_days)
        numpyro.deterministic("baseline_value", baseline)
        numpyro.deterministic("jitter_value", jitter)
        numpyro.deterministic("depth_ppm_approx", 1e6 * rp_rs**2)
        numpyro.sample("obs", dist.Normal(transit_flux + baseline, sigma), obs=flux)

    init_values = {
        "b": 0.6,
        "duration_days": duration_ref,
        "rp_rs": min(max(rp_ref, 0.006), 0.49),
    }
    if fixed_t0_days is None:
        init_values["t0_days"] = 0.0
    if fixed_limb is None:
        init_values["q1"] = 0.25
        init_values["q2"] = 0.3
    if fixed_baseline is None:
        init_values["baseline"] = 0.0
    if scatter_mode == "inferred":
        init_values["scatter"] = scatter_ref
    if fixed_jitter is None:
        init_values["jitter"] = median_flux_err

    fixed_values = {
        "t0_days": fixed_t0_days,
        "baseline": fixed_baseline,
        "jitter": fixed_jitter,
        "limb_darkening": fixed_limb,
        "scatter_prior_scale": scatter_prior_scale,
    }
    return model, init_values, rp_ref, duration_scale, jitter_scale, fixed_values


def infer_fixed_scatter(flux_err: np.ndarray, stat: str) -> float:
    flux_err = np.asarray(flux_err, dtype=float)
    good = np.isfinite(flux_err) & (flux_err > 0)
    if not np.any(good):
        raise RuntimeError("No positive finite flux_err values are available.")

    values = flux_err[good]
    if stat == "median":
        scatter = float(np.nanmedian(values))
    elif stat == "mean":
        scatter = float(np.nanmean(values))
    elif stat == "rms":
        scatter = float(np.sqrt(np.nanmean(values**2)))
    else:
        raise ValueError(f"Unknown fixed scatter statistic: {stat}")

    if not np.isfinite(scatter) or scatter <= 0:
        raise RuntimeError(f"Invalid inferred fixed scatter: {scatter}")
    return scatter


def effective_flux_err(args: argparse.Namespace, target: dict) -> tuple[np.ndarray, float]:
    if args.scatter_mode == "per-bin":
        flux_err = np.asarray(target["flux_err"], dtype=float)
        return flux_err, float(np.nanmedian(flux_err))

    if args.fixed_scatter_value is None:
        scatter = infer_fixed_scatter(target["flux_err"], args.fixed_scatter_stat)
    else:
        scatter = float(args.fixed_scatter_value)
        if not np.isfinite(scatter) or scatter <= 0:
            raise ValueError("--fixed-scatter-value must be positive and finite")

    if args.scatter_mode == "inferred":
        return np.asarray(target["flux_err"], dtype=float), scatter
    return np.full_like(target["flux_err"], scatter, dtype=float), scatter


def print_target_summary(target: dict, rp_ref: float) -> None:
    print("Target")
    print(f"  index:              {target['index']}")
    print(f"  name:               {target['name']}")
    print(f"  label:              {target['label']}")
    print(f"  period:             {target['period_days']:.15g} d (fixed)")
    print(f"  epoch:              {target['epoch_bkjd']:.15g} BKJD")
    print(f"  duration:           {target['duration_hours']:.15g} hr")
    print(f"  depth:              {target['depth_ppm']:.15g} ppm")
    print(f"  sqrt(depth):        {rp_ref:.6g} Rp/Rstar prior centre")
    print(f"  number of transits: {target['n_transits']}")
    print(f"  binned points:      {len(target['phase_time'])}")
    print(
        "  phase range:        "
        f"{np.min(target['phase_time']):.6g} .. {np.max(target['phase_time']):.6g} d"
    )
    print(
        "  median flux_err:    "
        f"{np.nanmedian(target['flux_err']):.6g}"
    )


def save_samples(output_dir: Path, target: dict, mcmc: MCMC) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{target['label']}_numpyro_mcmc.npz"
    samples = mcmc.get_samples(group_by_chain=True)
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)

    arrays = {name: np.asarray(value) for name, value in samples.items()}
    arrays.update(
        {f"extra_{name}": np.asarray(value) for name, value in extra_fields.items()}
    )
    arrays.update(
        {
            "name": np.asarray(target["name"]),
            "label": np.asarray(target["label"]),
            "period_days": np.asarray(target["period_days"]),
            "duration_days_ref": np.asarray(target["duration_days"]),
            "depth_ppm_ref": np.asarray(target["depth_ppm"]),
            "scatter_mode": np.asarray(target["scatter_mode"]),
            "scatter_value": np.asarray(target["scatter_value"]),
        }
    )
    np.savez(path, **arrays)
    return path


def write_corner_plot(path: Path, mcmc: MCMC, target: dict) -> bool:
    import corner
    import matplotlib.pyplot as plt

    samples = mcmc.get_samples()
    names = [
        name
        for name in (
            "b",
            "duration_days",
            "rp_rs",
            "t0_days",
            "q1",
            "q2",
            "baseline",
            "scatter",
            "jitter",
        )
        if name in samples
    ]
    if len(names) == 0:
        return False

    labels = {
        "b": "b",
        "duration_days": "duration [d]",
        "rp_rs": "Rp/Rstar",
        "t0_days": "t0 [d]",
        "q1": "q1",
        "q2": "q2",
        "baseline": "baseline",
        "scatter": "scatter",
        "jitter": "jitter",
    }
    values = np.column_stack([np.asarray(samples[name]) for name in names])
    if values.shape[0] <= values.shape[1]:
        print(
            "Skipping corner plot: need more posterior samples than "
            f"dimensions ({values.shape[0]} samples, {values.shape[1]} dimensions)."
        )
        return False

    levels = 1 - np.exp(-0.5 * np.array([1, 2]) ** 2)
    fig = corner.corner(
        values,
        labels=[labels[name] for name in names],
        smooth=1.0,
        levels=levels,
        hist_kwargs={"density": True},
    )
    fig.suptitle(target["name"], fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    if args.num_chains > 1 and args.chain_method == "parallel":
        numpyro.set_host_device_count(args.num_chains)

    target = load_target(args.library, args.index)
    model_flux_err, scatter_value = effective_flux_err(args, target)
    target["scatter_mode"] = args.scatter_mode
    target["scatter_value"] = scatter_value
    model, init_values, rp_ref, duration_scale, jitter_scale, fixed_values = make_model(
        args, target
    )
    print_target_summary(target, rp_ref)
    print("Priors")
    print(f"  duration scale:     {duration_scale:.6g} d")
    print(f"  Rp/Rstar scale:     {args.rp_prior_scale:.6g}")
    print(f"  scatter mode:       {args.scatter_mode}")
    if args.scatter_mode == "fixed":
        source = (
            f"{args.fixed_scatter_stat} flux_err"
            if args.fixed_scatter_value is None
            else "CLI value"
        )
        print(f"  fixed scatter:      {scatter_value:.6g} ({source})")
    elif args.scatter_mode == "inferred":
        source = (
            f"{args.fixed_scatter_stat} flux_err"
            if args.fixed_scatter_value is None
            else "CLI value"
        )
        print(f"  scatter reference:  {scatter_value:.6g} ({source})")
        print(f"  scatter prior:      HalfNormal({fixed_values['scatter_prior_scale']:.6g})")
    if fixed_values["t0_days"] is None:
        print(f"  t0 scale:           {args.t0_prior_scale:.6g} d")
    if fixed_values["jitter"] is None and args.scatter_mode != "inferred":
        print(f"  jitter scale:       {jitter_scale:.6g}")
    fixed_labels = []
    if fixed_values["t0_days"] is not None:
        fixed_labels.append(f"t0_days={fixed_values['t0_days']:.6g}")
    if fixed_values["baseline"] is not None:
        fixed_labels.append(f"baseline={fixed_values['baseline']:.6g}")
    if fixed_values["jitter"] is not None:
        fixed_labels.append(f"jitter={fixed_values['jitter']:.6g}")
    if fixed_values["limb_darkening"] is not None:
        u1, u2 = fixed_values["limb_darkening"]
        fixed_labels.append(f"u1={u1:.6g}, u2={u2:.6g}")
    if fixed_labels:
        print("Fixed")
        for label in fixed_labels:
            print(f"  {label}")

    kernel = NUTS(
        model,
        dense_mass=not args.diagonal_mass,
        target_accept_prob=args.target_accept_prob,
        max_tree_depth=args.max_tree_depth,
        init_strategy=init_to_value(values=init_values),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method=args.chain_method,
        progress_bar=not args.no_progress,
    )
    rng_key = jax.random.PRNGKey(args.seed)
    mcmc.run(
        rng_key,
        jnp.asarray(target["phase_time"]),
        flux=jnp.asarray(target["flux"]),
        flux_err=jnp.asarray(model_flux_err),
        extra_fields=("diverging", "accept_prob", "num_steps"),
    )

    print()
    mcmc.print_summary()

    extra_fields = mcmc.get_extra_fields()
    if "diverging" in extra_fields:
        diverging = np.asarray(extra_fields["diverging"], dtype=bool)
        print(f"Divergences: {int(np.sum(diverging))} / {diverging.size}")
    if "accept_prob" in extra_fields:
        accept_prob = np.asarray(extra_fields["accept_prob"], dtype=float)
        print(f"Mean accept_prob: {np.nanmean(accept_prob):.3f}")
    if "num_steps" in extra_fields:
        num_steps = np.asarray(extra_fields["num_steps"], dtype=float)
        print(f"Median NUTS steps: {np.nanmedian(num_steps):.0f}")

    if not args.no_corner:
        corner_path = args.corner_plot
        if corner_path is None:
            corner_path = args.plot_dir / f"{target['label']}_numpyro_mcmc_corner.png"
        if write_corner_plot(corner_path, mcmc, target):
            print(f"Saved {corner_path}")

    if not args.no_save:
        path = save_samples(args.output_dir, target, mcmc)
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
