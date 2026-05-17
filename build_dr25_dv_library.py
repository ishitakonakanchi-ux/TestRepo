#!/usr/bin/env python3
"""Build a library of good Kepler DR25 DV light curves for SBI training.

The script downloads official Kepler DR25 Data Validation time-series products
(`*_dvt.fits`) for TCEs listed in the NASA Exoplanet Archive DR25 TCE table.
For each TCE it extracts the official DV detrended curve (`1 + LC_DETREND`),
cuts a fixed window around transit, bins it to the SBI grid, and estimates
per-bin errors from the empirical scatter and the number of distinct transits.

By default this walks through the DR25 TCEs sorted by model SNR and keeps the
first 20 that pass the quality filters. Use `--all` for the full DR25 TCE table.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen, urlretrieve


MPLCONFIGDIR = Path("data/matplotlib-cache")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

try:
    import matplotlib
    import numpy as np
    from astropy.io import fits
except ModuleNotFoundError as err:
    missing = err.name
    raise SystemExit(
        f"Missing dependency '{missing}'. Run with the teaching venv, e.g.\n"
        "  /Users/rstiskalek/Projects/Teaching/venv_teach/bin/python "
        "build_dr25_dv_library.py"
    ) from err


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARCHIVE_API_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
)
TCE_TABLE = "q1_q17_dr25_tce"
KOI_TABLE = "q1_q17_dr25_koi"
DR25_DV_STAMP = "20160128150956"
SELECT_COLUMNS = [
    "kepid",
    "tce_plnt_num",
    "tce_period",
    "tce_time0bk",
    "tce_duration",
    "tce_depth",
    "tce_model_snr",
]
KOI_SELECT_COLUMNS = [
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_score",
    "koi_period",
    "koi_depth",
    "koi_prad",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
]


@dataclass(frozen=True)
class TCERecord:
    kepid: int
    tce_plnt_num: int
    period_days: float
    epoch_bkjd: float
    duration_hours: float
    depth_ppm: float
    model_snr: float

    @property
    def kic_string(self) -> str:
        return f"{self.kepid:09d}"

    @property
    def dv_filename(self) -> str:
        return f"kplr{self.kic_string}-{DR25_DV_STAMP}_dvt.fits"

    @property
    def dv_url(self) -> str:
        return (
            "https://archive.stsci.edu/missions/kepler/dv_files/"
            f"{self.kic_string[:4]}/{self.kic_string}/{self.dv_filename}"
        )

    @property
    def label(self) -> str:
        return f"kplr{self.kic_string}_tce{self.tce_plnt_num:02d}"

    @property
    def display_name(self) -> str:
        return f"KIC {self.kic_string} / TCE {self.tce_plnt_num:02d}"


@dataclass(frozen=True)
class KOIRecord:
    kepid: int
    kepoi_name: str
    kepler_name: str
    disposition: str
    pdisposition: str
    score: float
    period_days: float
    depth_ppm: float
    planet_radius_earth: float
    fpflag_nt: int
    fpflag_ss: int
    fpflag_co: int
    fpflag_ec: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and bin Kepler DR25 DV curves for SBI training."
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=20,
        help="Maximum number of TCE curves to process. Ignored with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every TCE returned by the DR25 table query.",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=None,
        help="Optional minimum DR25 model SNR filter.",
    )
    parser.add_argument(
        "--max-depth-ppm",
        type=float,
        default=None,
        help="Optional maximum DR25 fitted depth filter in ppm.",
    )
    parser.add_argument(
        "--allow-koi-false-positives",
        action="store_true",
        help="Keep TCEs whose matched KOI row is labelled FALSE POSITIVE.",
    )
    parser.add_argument(
        "--disable-odd-even-filter",
        action="store_true",
        help="Keep TCEs with a large odd/even transit-depth mismatch.",
    )
    parser.add_argument(
        "--odd-even-threshold",
        type=float,
        default=0.20,
        help=(
            "Reject if max odd/even binned mismatch inside transit exceeds "
            "this fraction of the DV model depth."
        ),
    )
    parser.add_argument(
        "--odd-even-min-abs",
        type=float,
        default=0.01,
        help="Minimum absolute odd/even mismatch required for rejection.",
    )
    parser.add_argument(
        "--disable-model-consistency-filter",
        action="store_true",
        help="Keep TCEs whose binned DV flux is inconsistent with the DV model.",
    )
    parser.add_argument(
        "--model-consistency-threshold",
        type=float,
        default=0.25,
        help=(
            "Reject if the maximum binned flux-model residual exceeds this "
            "fraction of the binned DV model depth."
        ),
    )
    parser.add_argument(
        "--model-consistency-min-abs",
        type=float,
        default=0.005,
        help="Minimum absolute binned flux-model residual required for rejection.",
    )
    parser.add_argument(
        "--dv-dir",
        type=Path,
        default=Path("data/dv"),
        help="Directory for cached official Kepler DR25 DV FITS files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dr25_dv_library"),
        help="Directory for binned curves, manifest, and training arrays.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("plots/dr25_dv_library_overview.png"),
        help="Overview plot of the binned DR25 DV curves.",
    )
    parser.add_argument(
        "--error-plot",
        type=Path,
        default=Path("plots/dr25_dv_library_errors.png"),
        help="Overview plot of the binned DR25 DV flux errors.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write the overview plot.",
    )
    parser.add_argument(
        "--max-plot-panels",
        type=int,
        default=20,
        help="Maximum number of successful TCEs to include in the overview plot.",
    )
    parser.add_argument(
        "--window-days",
        type=float,
        default=0.2,
        help="Minimum half-width of the transit window in days.",
    )
    parser.add_argument(
        "--window-mode",
        choices=("auto", "duration", "fixed"),
        default="auto",
        help=(
            "How to choose each half-window: auto from the DV model, "
            "duration from DR25 TDUR, or fixed from --window-days."
        ),
    )
    parser.add_argument(
        "--duration-window-factor",
        type=float,
        default=1.5,
        help=(
            "For --window-mode duration, or auto fallback, use this factor "
            "times the DR25 half-duration."
        ),
    )
    parser.add_argument(
        "--model-window-threshold",
        type=float,
        default=0.02,
        help=(
            "For --window-mode auto, define the model transit edge where the "
            "DV model is deeper than this fraction of its fitted depth."
        ),
    )
    parser.add_argument(
        "--model-window-padding",
        type=float,
        default=0.5,
        help=(
            "For --window-mode auto, add this fraction of the detected model "
            "half-width as baseline padding on each side."
        ),
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=50,
        help="Number of bins for the SBI grid.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only use already-cached DR25 DV FITS files.",
    )
    parser.add_argument(
        "--prune-dv-cache",
        action="store_true",
        help="Delete cached DR25 DV FITS files that are not in this selection.",
    )
    return parser.parse_args()


def parse_float(value: str) -> float:
    value = value.strip()
    if value == "":
        return np.nan
    return float(value)


def parse_int(value: str) -> int:
    return int(float(value.strip()))


def parse_optional_int(value: str) -> int:
    value = value.strip()
    if value == "":
        return 0
    return int(float(value))


def robust_mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    median = np.nanmedian(values)
    return 1.4826 * np.nanmedian(np.abs(values - median))


def format_bytes(n_bytes: int | float) -> str:
    n_bytes = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024.0 or unit == "GB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} GB"


def download_with_progress(url: str, path: Path) -> None:
    state = {"last_fraction": -1.0, "last_bytes": 0}
    bar_width = 24

    def reporthook(block_count: int, block_size: int, total_size: int) -> None:
        downloaded = block_count * block_size
        if total_size > 0:
            downloaded = min(downloaded, total_size)
            fraction = downloaded / total_size
            if fraction - state["last_fraction"] < 0.02 and fraction < 1.0:
                return
            state["last_fraction"] = fraction
            filled = int(round(bar_width * fraction))
            bar = "#" * filled + "-" * (bar_width - filled)
            message = (
                f"\r  download [{bar}] {format_bytes(downloaded)} / "
                f"{format_bytes(total_size)} ({100 * fraction:5.1f}%)"
            )
        else:
            if downloaded - state["last_bytes"] < 5 * 1024**2:
                return
            state["last_bytes"] = downloaded
            message = f"\r  download {format_bytes(downloaded)}"

        sys.stdout.write(message)
        sys.stdout.flush()
        if total_size > 0 and downloaded >= total_size:
            sys.stdout.write("\n")
            sys.stdout.flush()

    urlretrieve(url, path, reporthook=reporthook)
    if state["last_fraction"] < 1.0:
        sys.stdout.write("\n")
        sys.stdout.flush()


def build_tce_query_url(args: argparse.Namespace) -> str:
    where_clauses = []
    if args.min_snr is not None:
        where_clauses.append(f"tce_model_snr>{args.min_snr:g}")
    if args.max_depth_ppm is not None:
        where_clauses.append(f"tce_depth<{args.max_depth_ppm:g}")

    params = {
        "table": TCE_TABLE,
        "select": ",".join(SELECT_COLUMNS),
        "order": "tce_model_snr desc",
        "format": "csv",
    }
    if where_clauses:
        params["where"] = " and ".join(where_clauses)

    return ARCHIVE_API_URL + "?" + urlencode(params, safe=",><=", quote_via=quote)


def build_koi_query_url() -> str:
    params = {
        "table": KOI_TABLE,
        "select": ",".join(KOI_SELECT_COLUMNS),
        "format": "csv",
    }
    return ARCHIVE_API_URL + "?" + urlencode(params, safe=",", quote_via=quote)


def fetch_tce_catalog(args: argparse.Namespace) -> list[TCERecord]:
    url = build_tce_query_url(args)
    request = Request(url, headers={"User-Agent": "IshitaRepo DR25 DV downloader"})
    with urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8")

    rows = csv.DictReader(io.StringIO(text))
    records: list[TCERecord] = []
    for row in rows:
        try:
            record = TCERecord(
                kepid=parse_int(row["kepid"]),
                tce_plnt_num=parse_int(row["tce_plnt_num"]),
                period_days=parse_float(row["tce_period"]),
                epoch_bkjd=parse_float(row["tce_time0bk"]),
                duration_hours=parse_float(row["tce_duration"]),
                depth_ppm=parse_float(row["tce_depth"]),
                model_snr=parse_float(row["tce_model_snr"]),
            )
        except (KeyError, ValueError):
            continue

        if not np.isfinite(record.period_days):
            continue
        if not np.isfinite(record.epoch_bkjd):
            continue
        records.append(record)

    return records


def fetch_koi_catalog() -> dict[int, list[KOIRecord]]:
    url = build_koi_query_url()
    request = Request(url, headers={"User-Agent": "IshitaRepo DR25 DV downloader"})
    with urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8")

    rows = csv.DictReader(io.StringIO(text))
    by_kepid: dict[int, list[KOIRecord]] = {}
    for row in rows:
        try:
            record = KOIRecord(
                kepid=parse_int(row["kepid"]),
                kepoi_name=row.get("kepoi_name", "").strip(),
                kepler_name=row.get("kepler_name", "").strip(),
                disposition=row.get("koi_disposition", "").strip(),
                pdisposition=row.get("koi_pdisposition", "").strip(),
                score=parse_float(row.get("koi_score", "")),
                period_days=parse_float(row.get("koi_period", "")),
                depth_ppm=parse_float(row.get("koi_depth", "")),
                planet_radius_earth=parse_float(row.get("koi_prad", "")),
                fpflag_nt=parse_optional_int(row.get("koi_fpflag_nt", "")),
                fpflag_ss=parse_optional_int(row.get("koi_fpflag_ss", "")),
                fpflag_co=parse_optional_int(row.get("koi_fpflag_co", "")),
                fpflag_ec=parse_optional_int(row.get("koi_fpflag_ec", "")),
            )
        except (KeyError, ValueError):
            continue
        by_kepid.setdefault(record.kepid, []).append(record)
    return by_kepid


def koi_period_match_score(record: TCERecord, koi: KOIRecord) -> float:
    if not np.isfinite(koi.period_days) or koi.period_days <= 0:
        return float("inf")
    candidates = [
        abs(koi.period_days - record.period_days) / record.period_days,
        abs(koi.period_days - 2.0 * record.period_days) / (2.0 * record.period_days),
        abs(2.0 * koi.period_days - record.period_days) / record.period_days,
    ]
    return min(candidates)


def match_koi(record: TCERecord, koi_by_kepid: dict[int, list[KOIRecord]]) -> KOIRecord | None:
    matches = koi_by_kepid.get(record.kepid, [])
    if not matches:
        return None
    return min(matches, key=lambda koi: koi_period_match_score(record, koi))


def koi_rejection_reason(koi: KOIRecord | None, allow_false_positives: bool) -> str:
    if koi is None or allow_false_positives:
        return ""
    disposition = koi.disposition.upper()
    pdisposition = koi.pdisposition.upper()
    if disposition == "FALSE POSITIVE" or pdisposition == "FALSE POSITIVE":
        return f"matched KOI {koi.kepoi_name} is FALSE POSITIVE"
    if koi.fpflag_ss:
        return f"matched KOI {koi.kepoi_name} has significant-secondary flag"
    return ""


def write_selected_catalog(path: Path, records: list[TCERecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "kepid",
        "tce_plnt_num",
        "period_days",
        "epoch_bkjd",
        "duration_hours",
        "depth_ppm",
        "model_snr",
        "dv_filename",
        "dv_url",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "kepid": record.kepid,
                    "name": record.display_name,
                    "tce_plnt_num": record.tce_plnt_num,
                    "period_days": record.period_days,
                    "epoch_bkjd": record.epoch_bkjd,
                    "duration_hours": record.duration_hours,
                    "depth_ppm": record.depth_ppm,
                    "model_snr": record.model_snr,
                    "dv_filename": record.dv_filename,
                    "dv_url": record.dv_url,
                }
            )


def ensure_dv_file(record: TCERecord, dv_dir: Path, no_download: bool) -> Path:
    dv_dir.mkdir(parents=True, exist_ok=True)
    path = dv_dir / record.dv_filename
    if path.exists() and path.stat().st_size > 0:
        print(f"  using cached FITS ({format_bytes(path.stat().st_size)})", flush=True)
        return path
    if no_download:
        raise FileNotFoundError(f"{path} is not cached")

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"Downloading {record.dv_filename}", flush=True)
    try:
        download_with_progress(record.dv_url, tmp_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    tmp_path.replace(path)
    print(f"  cached FITS ({format_bytes(path.stat().st_size)})", flush=True)
    return path


def clean_stale_outputs(
    curves_dir: Path,
    dv_dir: Path,
    selected_records: list[TCERecord],
    prune_dv_cache: bool,
) -> None:
    expected_curves = {
        curves_dir / f"{record.label}_sbi_grid.csv" for record in selected_records
    }
    if curves_dir.exists():
        for path in curves_dir.glob("*_sbi_grid.csv"):
            if path not in expected_curves:
                path.unlink()

    if dv_dir.exists():
        for path in dv_dir.glob("*.tmp"):
            path.unlink()

    if prune_dv_cache:
        expected_fits = {record.dv_filename for record in selected_records}
        if dv_dir.exists():
            for path in dv_dir.glob(f"kplr*-{DR25_DV_STAMP}_dvt.fits"):
                if path.name not in expected_fits:
                    path.unlink()


def duration_based_window(
    duration_hours: float,
    minimum_window_days: float,
    duration_window_factor: float,
) -> float:
    if duration_window_factor <= 0:
        return minimum_window_days
    half_duration_days = 0.5 * duration_hours / 24.0
    return max(minimum_window_days, duration_window_factor * half_duration_days)


def model_based_window(
    phase_time: np.ndarray,
    model: np.ndarray,
    minimum_window_days: float,
    threshold_fraction: float,
    padding_fraction: float,
) -> float | None:
    finite = np.isfinite(phase_time) & np.isfinite(model)
    if np.sum(finite) < 2:
        return None

    phase = phase_time[finite]
    model_flux = model[finite]
    baseline = float(np.nanpercentile(model_flux, 95))
    depth = baseline - float(np.nanmin(model_flux))
    if not np.isfinite(depth) or depth <= 0:
        return None

    threshold = baseline - threshold_fraction * depth
    in_transit = model_flux < threshold
    if np.sum(in_transit) < 2:
        return None

    left = float(np.nanmin(phase[in_transit]))
    right = float(np.nanmax(phase[in_transit]))
    half_width = max(abs(left), abs(right))
    half_span = 0.5 * max(right - left, 0.0)
    window_days = half_width + padding_fraction * half_span
    if not np.isfinite(window_days) or window_days <= 0:
        return None
    return max(minimum_window_days, window_days)


def choose_window_days(
    phase_time: np.ndarray,
    model: np.ndarray,
    duration_hours: float,
    minimum_window_days: float,
    window_mode: str,
    duration_window_factor: float,
    model_window_threshold: float,
    model_window_padding: float,
) -> float:
    if window_mode == "fixed":
        return minimum_window_days
    if window_mode == "duration":
        return duration_based_window(
            duration_hours, minimum_window_days, duration_window_factor
        )

    model_window = model_based_window(
        phase_time,
        model,
        minimum_window_days,
        model_window_threshold,
        model_window_padding,
    )
    if model_window is not None:
        return model_window
    return duration_based_window(duration_hours, minimum_window_days, duration_window_factor)


def choose_tce_hdu(hdul, record: TCERecord) -> int:
    wanted = f"TCE_{record.tce_plnt_num}"
    for index, hdu in enumerate(hdul):
        if hdu.name == wanted:
            return index

    best_index = None
    best_delta = float("inf")
    for index, hdu in enumerate(hdul):
        if not hdu.name.startswith("TCE_"):
            continue
        period = hdu.header.get("TPERIOD")
        if period is None:
            continue
        delta = abs(float(period) - record.period_days)
        if delta < best_delta:
            best_index = index
            best_delta = delta

    if best_index is None:
        raise RuntimeError("No TCE extension found in DV FITS file")
    return best_index


def read_dv_curve(
    path: Path,
    record: TCERecord,
    minimum_window_days: float,
    window_mode: str,
    duration_window_factor: float,
    model_window_threshold: float,
    model_window_padding: float,
) -> dict:
    with fits.open(path) as hdul:
        hdu_index = choose_tce_hdu(hdul, record)
        hdu = hdul[hdu_index]
        table = hdu.data
        header = hdu.header

        time = np.asarray(table["TIME"], dtype=float)
        phase_time = np.asarray(table["PHASE"], dtype=float)
        flux = 1.0 + np.asarray(table["LC_DETREND"], dtype=float)
        model = 1.0 + np.asarray(table["MODEL_INIT"], dtype=float)

        period_days = float(header.get("TPERIOD", record.period_days))
        epoch_bkjd = float(header.get("TEPOCH", record.epoch_bkjd))
        duration_hours = float(header.get("TDUR", record.duration_hours))
        depth_ppm = float(header.get("TDEPTH", record.depth_ppm))
        n_transits = int(header.get("NTRANS", 0))
        tce_name = hdu.name

    window_days = choose_window_days(
        phase_time,
        model,
        duration_hours,
        minimum_window_days,
        window_mode,
        duration_window_factor,
        model_window_threshold,
        model_window_padding,
    )
    keep = (
        np.isfinite(time)
        & np.isfinite(phase_time)
        & np.isfinite(flux)
        & np.isfinite(model)
        & (np.abs(phase_time) <= window_days)
    )
    if not np.any(keep):
        raise RuntimeError("No finite DV points in the requested transit window")

    order = np.argsort(phase_time[keep])
    time = time[keep][order]
    phase_time = phase_time[keep][order]
    flux = flux[keep][order]
    model = model[keep][order]

    return {
        "time": time,
        "phase_time": phase_time,
        "flux": flux,
        "model": model,
        "epoch_index": np.rint((time - epoch_bkjd) / period_days).astype(int),
        "tce_name": tce_name,
        "period_days": period_days,
        "epoch_bkjd": epoch_bkjd,
        "duration_hours": duration_hours,
        "depth_ppm": depth_ppm,
        "n_transits": n_transits,
        "window_days": window_days,
        "window_mode": window_mode,
    }


def bin_curve(dv: dict, window_days: float, n_bins: int) -> dict[str, np.ndarray]:
    edges = np.linspace(-window_days, window_days, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned_flux = np.full(n_bins, np.nan)
    binned_model = np.full(n_bins, np.nan)
    flux_scatter = np.full(n_bins, np.nan)
    flux_err = np.full(n_bins, np.nan)
    flux_err_n_points = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    n_eff = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        in_bin = (edges[i] <= dv["phase_time"]) & (dv["phase_time"] < edges[i + 1])
        if i == n_bins - 1:
            in_bin |= dv["phase_time"] == edges[i + 1]

        finite_flux = in_bin & np.isfinite(dv["flux"])
        finite_model = in_bin & np.isfinite(dv["model"])
        flux_values = dv["flux"][finite_flux]
        model_values = dv["model"][finite_model]
        epoch_values = dv["epoch_index"][finite_flux]

        counts[i] = len(flux_values)
        if counts[i] == 0:
            continue
        n_eff[i] = len(np.unique(epoch_values))

        binned_flux[i] = np.nanmedian(flux_values)
        if len(model_values) > 0:
            binned_model[i] = np.nanmedian(model_values)

        scatter = robust_mad(flux_values)
        if np.isfinite(scatter) and scatter > 0:
            flux_scatter[i] = scatter
            if n_eff[i] > 0:
                flux_err[i] = 1.253 * scatter / np.sqrt(n_eff[i])
            flux_err_n_points[i] = scatter / np.sqrt(counts[i])

    good = np.isfinite(binned_flux)
    if not np.any(good):
        raise RuntimeError("No populated bins in the transit window")
    binned_flux[~good] = np.interp(centers[~good], centers[good], binned_flux[good])

    good_model = np.isfinite(binned_model)
    if np.any(good_model):
        binned_model[~good_model] = np.interp(
            centers[~good_model], centers[good_model], binned_model[good_model]
        )

    for values in (flux_err, flux_err_n_points, flux_scatter):
        good_values = np.isfinite(values)
        fill_value = np.nanmedian(values[good_values]) if np.any(good_values) else np.nan
        values[~good_values] = fill_value

    return {
        "phase_time": centers,
        "flux": binned_flux,
        "model": binned_model,
        "flux_err": flux_err,
        "flux_err_n_points": flux_err_n_points,
        "flux_scatter": flux_scatter,
        "n_points": counts,
        "n_eff": n_eff,
    }


def odd_even_metrics(dv: dict, binned: dict, window_days: float, n_bins: int) -> dict:
    edges = np.linspace(-window_days, window_days, n_bins + 1)
    even_flux = np.full(n_bins, np.nan)
    odd_flux = np.full(n_bins, np.nan)
    even_counts = np.zeros(n_bins, dtype=int)
    odd_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        in_bin = (edges[i] <= dv["phase_time"]) & (dv["phase_time"] < edges[i + 1])
        if i == n_bins - 1:
            in_bin |= dv["phase_time"] == edges[i + 1]

        finite_flux = in_bin & np.isfinite(dv["flux"])
        even = finite_flux & (dv["epoch_index"] % 2 == 0)
        odd = finite_flux & (dv["epoch_index"] % 2 == 1)
        even_counts[i] = int(np.sum(even))
        odd_counts[i] = int(np.sum(odd))
        if even_counts[i] > 0:
            even_flux[i] = np.nanmedian(dv["flux"][even])
        if odd_counts[i] > 0:
            odd_flux[i] = np.nanmedian(dv["flux"][odd])

    model = binned["model"]
    finite_model = model[np.isfinite(model)]
    if len(finite_model) == 0:
        return {
            "odd_even_metric": np.nan,
            "odd_even_max_abs": np.nan,
            "odd_even_depth": np.nan,
            "odd_even_n_bins": 0,
        }
    baseline = float(np.nanpercentile(finite_model, 95))
    depth = baseline - float(np.nanmin(finite_model))
    if not np.isfinite(depth) or depth <= 0:
        return {
            "odd_even_metric": np.nan,
            "odd_even_max_abs": np.nan,
            "odd_even_depth": np.nan,
            "odd_even_n_bins": 0,
        }

    in_transit = model < baseline - 0.20 * depth
    comparable = (
        in_transit
        & np.isfinite(even_flux)
        & np.isfinite(odd_flux)
        & (even_counts >= 3)
        & (odd_counts >= 3)
    )
    if not np.any(comparable):
        return {
            "odd_even_metric": np.nan,
            "odd_even_max_abs": np.nan,
            "odd_even_depth": depth,
            "odd_even_n_bins": 0,
        }

    max_abs = float(np.nanmax(np.abs(even_flux[comparable] - odd_flux[comparable])))
    return {
        "odd_even_metric": max_abs / depth,
        "odd_even_max_abs": max_abs,
        "odd_even_depth": depth,
        "odd_even_n_bins": int(np.sum(comparable)),
    }


def model_consistency_metrics(binned: dict) -> dict:
    flux = binned["flux"]
    model = binned["model"]
    finite = np.isfinite(flux) & np.isfinite(model)
    if not np.any(finite):
        return {
            "model_resid_metric": np.nan,
            "model_resid_median_metric": np.nan,
            "model_resid_max_abs": np.nan,
            "model_resid_depth": np.nan,
        }

    model_flux = model[finite]
    baseline = float(np.nanpercentile(model_flux, 95))
    depth = baseline - float(np.nanmin(model_flux))
    if not np.isfinite(depth) or depth <= 0:
        return {
            "model_resid_metric": np.nan,
            "model_resid_median_metric": np.nan,
            "model_resid_max_abs": np.nan,
            "model_resid_depth": depth,
        }

    abs_residual = np.abs(flux[finite] - model[finite])
    max_abs = float(np.nanmax(abs_residual))
    median_abs = float(np.nanmedian(abs_residual))
    return {
        "model_resid_metric": max_abs / depth,
        "model_resid_median_metric": median_abs / depth,
        "model_resid_max_abs": max_abs,
        "model_resid_depth": depth,
    }


def model_consistency_rejection_reason(
    metrics: dict,
    threshold: float,
    min_abs: float,
) -> str:
    metric = metrics.get("model_resid_metric", np.nan)
    max_abs = metrics.get("model_resid_max_abs", np.nan)
    if not np.isfinite(metric) or not np.isfinite(max_abs):
        return ""
    if metric > threshold and max_abs > min_abs:
        return (
            f"model residual {metric:.2f} depth fractions "
            f"({max_abs:.3g} absolute)"
        )
    return ""


def odd_even_rejection_reason(
    metrics: dict,
    threshold: float,
    min_abs: float,
) -> str:
    metric = metrics.get("odd_even_metric", np.nan)
    max_abs = metrics.get("odd_even_max_abs", np.nan)
    if not np.isfinite(metric) or not np.isfinite(max_abs):
        return ""
    if metric > threshold and max_abs > min_abs:
        return (
            f"odd/even mismatch {metric:.2f} depth fractions "
            f"({max_abs:.3g} absolute)"
        )
    return ""


def write_binned_curve(path: Path, record: TCERecord, dv: dict, binned: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "kepid",
        "tce_plnt_num",
        "phase_time",
        "flux",
        "model",
        "flux_err",
        "flux_err_n_points",
        "flux_scatter",
        "n_points",
        "n_eff",
        "period_days",
        "epoch_bkjd",
        "duration_hours",
        "depth_ppm",
        "n_transits",
        "tce_name",
        "window_days",
        "window_mode",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(binned["phase_time"])):
            writer.writerow(
                {
                    "kepid": record.kepid,
                    "name": record.display_name,
                    "tce_plnt_num": record.tce_plnt_num,
                    "phase_time": binned["phase_time"][i],
                    "flux": binned["flux"][i],
                    "model": binned["model"][i],
                    "flux_err": binned["flux_err"][i],
                    "flux_err_n_points": binned["flux_err_n_points"][i],
                    "flux_scatter": binned["flux_scatter"][i],
                    "n_points": binned["n_points"][i],
                    "n_eff": binned["n_eff"][i],
                    "period_days": dv["period_days"],
                    "epoch_bkjd": dv["epoch_bkjd"],
                    "duration_hours": dv["duration_hours"],
                    "depth_ppm": dv["depth_ppm"],
                    "n_transits": dv["n_transits"],
                    "tce_name": dv["tce_name"],
                    "window_days": dv["window_days"],
                    "window_mode": dv["window_mode"],
                }
            )


def manifest_row(
    record: TCERecord,
    status: str,
    fits_path: Path | None = None,
    curve_path: Path | None = None,
    koi: KOIRecord | None = None,
    dv: dict | None = None,
    binned: dict | None = None,
    metrics: dict | None = None,
    reject_reason: str = "",
    error: str = "",
) -> dict:
    row = {
        "name": record.display_name,
        "kepid": record.kepid,
        "tce_plnt_num": record.tce_plnt_num,
        "status": status,
        "reject_reason": reject_reason,
        "error": error,
        "fits_path": str(fits_path) if fits_path else "",
        "curve_path": str(curve_path) if curve_path else "",
        "koi_name": "",
        "koi_disposition": "",
        "koi_pdisposition": "",
        "koi_period_days": "",
        "koi_score": "",
        "koi_fpflag_ss": "",
        "catalog_period_days": record.period_days,
        "catalog_epoch_bkjd": record.epoch_bkjd,
        "catalog_duration_hours": record.duration_hours,
        "catalog_depth_ppm": record.depth_ppm,
        "catalog_model_snr": record.model_snr,
        "dv_tce_name": "",
        "dv_period_days": "",
        "dv_epoch_bkjd": "",
        "dv_duration_hours": "",
        "dv_depth_ppm": "",
        "dv_n_transits": "",
        "window_days": "",
        "window_mode": "",
        "n_dv_points": "",
        "n_populated_bins": "",
        "median_flux_err": "",
        "median_flux_scatter": "",
        "median_n_eff": "",
        "model_resid_metric": "",
        "model_resid_median_metric": "",
        "model_resid_max_abs": "",
        "model_resid_depth": "",
        "odd_even_metric": "",
        "odd_even_max_abs": "",
        "odd_even_depth": "",
        "odd_even_n_bins": "",
    }
    if koi is not None:
        row.update(
            {
                "koi_name": koi.kepoi_name,
                "koi_disposition": koi.disposition,
                "koi_pdisposition": koi.pdisposition,
                "koi_period_days": koi.period_days,
                "koi_score": koi.score,
                "koi_fpflag_ss": koi.fpflag_ss,
            }
        )
    if dv is not None:
        row.update(
            {
                "dv_tce_name": dv["tce_name"],
                "dv_period_days": dv["period_days"],
                "dv_epoch_bkjd": dv["epoch_bkjd"],
                "dv_duration_hours": dv["duration_hours"],
                "dv_depth_ppm": dv["depth_ppm"],
                "dv_n_transits": dv["n_transits"],
                "window_days": dv["window_days"],
                "window_mode": dv["window_mode"],
                "n_dv_points": len(dv["phase_time"]),
            }
        )
    if binned is not None:
        row.update(
            {
                "n_populated_bins": int(np.sum(binned["n_points"] > 0)),
                "median_flux_err": float(np.nanmedian(binned["flux_err"])),
                "median_flux_scatter": float(np.nanmedian(binned["flux_scatter"])),
                "median_n_eff": float(np.nanmedian(binned["n_eff"])),
            }
        )
    if metrics is not None:
        row.update(
            {
                "model_resid_metric": metrics.get("model_resid_metric", ""),
                "model_resid_median_metric": metrics.get(
                    "model_resid_median_metric", ""
                ),
                "model_resid_max_abs": metrics.get("model_resid_max_abs", ""),
                "model_resid_depth": metrics.get("model_resid_depth", ""),
                "odd_even_metric": metrics.get("odd_even_metric", ""),
                "odd_even_max_abs": metrics.get("odd_even_max_abs", ""),
                "odd_even_depth": metrics.get("odd_even_depth", ""),
                "odd_even_n_bins": metrics.get("odd_even_n_bins", ""),
            }
        )
    return row


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_npz_library(path: Path, successes: list[tuple[TCERecord, dict, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not successes:
        return

    records = [item[0] for item in successes]
    dv_curves = [item[1] for item in successes]
    binned_curves = [item[2] for item in successes]
    np.savez_compressed(
        path,
        phase_time=np.stack([curve["phase_time"] for curve in binned_curves]),
        flux=np.stack([curve["flux"] for curve in binned_curves]),
        model=np.stack([curve["model"] for curve in binned_curves]),
        flux_err=np.stack([curve["flux_err"] for curve in binned_curves]),
        flux_err_n_points=np.stack(
            [curve["flux_err_n_points"] for curve in binned_curves]
        ),
        flux_scatter=np.stack([curve["flux_scatter"] for curve in binned_curves]),
        n_points=np.stack([curve["n_points"] for curve in binned_curves]),
        n_eff=np.stack([curve["n_eff"] for curve in binned_curves]),
        kepid=np.array([record.kepid for record in records], dtype=int),
        name=np.array([record.display_name for record in records]),
        label=np.array([record.label for record in records]),
        tce_plnt_num=np.array([record.tce_plnt_num for record in records], dtype=int),
        catalog_period_days=np.array([record.period_days for record in records]),
        catalog_epoch_bkjd=np.array([record.epoch_bkjd for record in records]),
        catalog_duration_hours=np.array([record.duration_hours for record in records]),
        catalog_depth_ppm=np.array([record.depth_ppm for record in records]),
        catalog_model_snr=np.array([record.model_snr for record in records]),
        dv_period_days=np.array([dv["period_days"] for dv in dv_curves]),
        dv_epoch_bkjd=np.array([dv["epoch_bkjd"] for dv in dv_curves]),
        dv_duration_hours=np.array([dv["duration_hours"] for dv in dv_curves]),
        dv_depth_ppm=np.array([dv["depth_ppm"] for dv in dv_curves]),
        dv_n_transits=np.array([dv["n_transits"] for dv in dv_curves], dtype=int),
        window_days=np.array([dv["window_days"] for dv in dv_curves]),
        window_mode=np.array([dv["window_mode"] for dv in dv_curves]),
        odd_even_metric=np.array(
            [curve.get("odd_even_metric", np.nan) for curve in binned_curves]
        ),
        odd_even_max_abs=np.array(
            [curve.get("odd_even_max_abs", np.nan) for curve in binned_curves]
        ),
        model_resid_metric=np.array(
            [curve.get("model_resid_metric", np.nan) for curve in binned_curves]
        ),
        model_resid_median_metric=np.array(
            [curve.get("model_resid_median_metric", np.nan) for curve in binned_curves]
        ),
        model_resid_max_abs=np.array(
            [curve.get("model_resid_max_abs", np.nan) for curve in binned_curves]
        ),
    )


def plot_library_overview(
    path: Path,
    successes: list[tuple[TCERecord, dict, dict]],
    max_panels: int,
) -> None:
    if not successes or max_panels <= 0:
        return

    selected = successes[:max_panels]
    n_panels = len(selected)
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 2.55 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for panel_index, (ax, (record, dv, binned)) in enumerate(zip(axes, selected)):
        ax.plot(
            binned["phase_time"],
            binned["model"],
            color="C1",
            lw=1.1,
            label="DV model" if panel_index == 0 else None,
        )
        ax.fill_between(
            binned["phase_time"],
            binned["flux"] - binned["flux_err"],
            binned["flux"] + binned["flux_err"],
            color="C0",
            alpha=0.16,
            lw=0,
        )
        ax.errorbar(
            binned["phase_time"],
            binned["flux"],
            yerr=binned["flux_err"],
            fmt="o",
            ms=2.4,
            lw=0.8,
            color="C0",
            ecolor="0.12",
            elinewidth=0.9,
            capsize=1.5,
            capthick=0.7,
            label="binned DV flux" if panel_index == 0 else None,
        )
        half_duration = 0.5 * dv["duration_hours"] / 24.0
        ax.axvspan(-half_duration, half_duration, color="C3", alpha=0.08)
        ax.axhline(1.0, color="0.25", lw=0.6, alpha=0.55)
        ax.set_xlim(-dv["window_days"], dv["window_days"])
        finite_y = np.concatenate(
            [
                binned["flux"][np.isfinite(binned["flux"])],
                binned["model"][np.isfinite(binned["model"])],
            ]
        )
        if len(finite_y) > 0:
            y_min = float(np.nanmin(finite_y))
            y_max = float(np.nanmax(finite_y))
            padding = max(0.015, 0.08 * (y_max - y_min))
            ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_title(
            f"{record.display_name}\n"
            f"P={dv['period_days']:.4g} d, depth={dv['depth_ppm']:.0f} ppm",
            fontsize=8,
        )
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=7)
        if panel_index == 0:
            ax.legend(loc="best", fontsize=7, framealpha=0.85)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    for i, ax in enumerate(axes[:n_panels]):
        row = i // n_cols
        col = i % n_cols
        if row == n_rows - 1:
            ax.set_xlabel("Time from transit [d]", fontsize=8)
        if col == 0:
            ax.set_ylabel("Relative flux", fontsize=8)

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_overview(
    path: Path,
    successes: list[tuple[TCERecord, dict, dict]],
    max_panels: int,
) -> None:
    if not successes or max_panels <= 0:
        return

    selected = successes[:max_panels]
    n_panels = len(selected)
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 2.2 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (record, dv, binned) in zip(axes, selected):
        ax.plot(
            binned["phase_time"],
            binned["flux_err"],
            marker="o",
            ms=2.2,
            lw=0.9,
            color="C3",
            label="flux_err",
        )
        ax.set_yscale("log")
        ax.set_xlim(-dv["window_days"], dv["window_days"])
        ax.set_title(
            f"{record.display_name}\n"
            f"median err={np.nanmedian(binned['flux_err']):.2g}, "
            f"median n_eff={np.nanmedian(binned['n_eff']):.0f}",
            fontsize=8,
        )
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=7)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    for i, ax in enumerate(axes[:n_panels]):
        row = i // n_cols
        col = i % n_cols
        if row == n_rows - 1:
            ax.set_xlabel("Time from transit [d]", fontsize=8)
        if col == 0:
            ax.set_ylabel("Binned flux error", fontsize=8)

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.max_targets <= 0 and not args.all:
        raise ValueError("--max-targets must be positive unless --all is used")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = args.output_dir / "curves"

    print("Querying NASA Exoplanet Archive DR25 TCE table", flush=True)
    records = fetch_tce_catalog(args)
    if not records:
        raise RuntimeError("No DR25 TCE records matched the query")

    koi_by_kepid: dict[int, list[KOIRecord]] = {}
    if not args.allow_koi_false_positives:
        print("Querying NASA Exoplanet Archive DR25 KOI table", flush=True)
        koi_by_kepid = fetch_koi_catalog()

    manifest_rows: list[dict] = []
    successes: list[tuple[TCERecord, dict, dict]] = []
    target_successes = len(records) if args.all else args.max_targets

    for index, record in enumerate(records, start=1):
        if not args.all and len(successes) >= target_successes:
            break

        print(
            f"[{index}/{len(records)}; accepted {len(successes)}/{target_successes}] "
            f"{record.label}: "
            f"SNR={record.model_snr:.1f}, depth={record.depth_ppm:.1f} ppm",
            flush=True,
        )
        fits_path: Path | None = None
        curve_path = curves_dir / f"{record.label}_sbi_grid.csv"
        koi = match_koi(record, koi_by_kepid)
        reject_reason = koi_rejection_reason(koi, args.allow_koi_false_positives)
        if reject_reason:
            manifest_rows.append(
                manifest_row(
                    record,
                    status="rejected",
                    curve_path=curve_path,
                    koi=koi,
                    reject_reason=reject_reason,
                )
            )
            print(f"  rejected: {reject_reason}", flush=True)
            continue

        try:
            fits_path = ensure_dv_file(record, args.dv_dir, args.no_download)
            dv = read_dv_curve(
                fits_path,
                record,
                args.window_days,
                args.window_mode,
                args.duration_window_factor,
                args.model_window_threshold,
                args.model_window_padding,
            )
            if dv["window_days"] > args.window_days:
                print(
                    f"  expanded half-window to {dv['window_days']:.3f} d",
                    flush=True,
                )
            binned = bin_curve(dv, dv["window_days"], args.n_bins)
            metrics = model_consistency_metrics(binned)
            binned.update(metrics)
            if not args.disable_model_consistency_filter:
                reject_reason = model_consistency_rejection_reason(
                    metrics,
                    args.model_consistency_threshold,
                    args.model_consistency_min_abs,
                )
                if reject_reason:
                    manifest_rows.append(
                        manifest_row(
                            record,
                            status="rejected",
                            fits_path=fits_path,
                            curve_path=curve_path,
                            koi=koi,
                            dv=dv,
                            binned=binned,
                            metrics=metrics,
                            reject_reason=reject_reason,
                        )
                    )
                    print(f"  rejected: {reject_reason}", flush=True)
                    continue

            metrics.update(odd_even_metrics(dv, binned, dv["window_days"], args.n_bins))
            binned.update(metrics)
            if not args.disable_odd_even_filter:
                reject_reason = odd_even_rejection_reason(
                    metrics,
                    args.odd_even_threshold,
                    args.odd_even_min_abs,
                )
                if reject_reason:
                    manifest_rows.append(
                        manifest_row(
                            record,
                            status="rejected",
                            fits_path=fits_path,
                            curve_path=curve_path,
                            koi=koi,
                            dv=dv,
                            binned=binned,
                            metrics=metrics,
                            reject_reason=reject_reason,
                        )
                    )
                    print(f"  rejected: {reject_reason}", flush=True)
                    continue

            write_binned_curve(curve_path, record, dv, binned)
        except Exception as err:
            manifest_rows.append(
                manifest_row(
                    record,
                    status="failed",
                    fits_path=fits_path,
                    curve_path=curve_path,
                    koi=koi,
                    error=str(err),
                )
            )
            print(f"  failed: {err}", file=sys.stderr, flush=True)
            continue

        successes.append((record, dv, binned))
        manifest_rows.append(
            manifest_row(
                record,
                status="ok",
                fits_path=fits_path,
                curve_path=curve_path,
                koi=koi,
                dv=dv,
                binned=binned,
                metrics=metrics,
            )
        )
        print(
            f"  saved {curve_path} "
            f"({len(dv['phase_time'])} DV points, median n_eff "
            f"{np.nanmedian(binned['n_eff']):.0f})",
            flush=True,
        )

    if not successes:
        raise RuntimeError("No TCEs were accepted into the DR25 DV library")

    selected_records = [item[0] for item in successes]
    selected_path = args.output_dir / "selected_tces.csv"
    write_selected_catalog(selected_path, selected_records)
    clean_stale_outputs(
        curves_dir,
        args.dv_dir,
        selected_records,
        args.prune_dv_cache,
    )

    manifest_path = args.output_dir / "manifest.csv"
    npz_path = args.output_dir / "dr25_dv_sbi_library.npz"
    write_manifest(manifest_path, manifest_rows)
    write_npz_library(npz_path, successes)
    if not args.no_plot:
        plot_library_overview(args.plot, successes, args.max_plot_panels)
        plot_error_overview(args.error_plot, successes, args.max_plot_panels)

    n_tried = len(manifest_rows)
    n_rejected = sum(row["status"] == "rejected" for row in manifest_rows)
    n_failed = sum(row["status"] == "failed" for row in manifest_rows)
    print(f"Selected {len(selected_records)} accepted TCEs", flush=True)
    print(
        f"Tried {n_tried} TCEs: {len(selected_records)} accepted, "
        f"{n_rejected} rejected, {n_failed} failed",
        flush=True,
    )
    print(f"Saved {selected_path}", flush=True)
    print(f"Saved {manifest_path}", flush=True)
    if successes:
        print(f"Saved {npz_path}", flush=True)
    if successes and not args.no_plot:
        print(f"Saved {args.plot}", flush=True)
        print(f"Saved {args.error_plot}", flush=True)
    print(f"Catalog rows available: {len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
