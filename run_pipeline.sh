#!/bin/bash
# Build the Kepler DR25 library, train a noise-aware amortized NPE, and
# compare it with MCMC on two mocks and one or more Kepler targets.
#
# The library MUST be binned on the same fixed grid the simulator uses (t_obs in
# transit_sbi.py): N_OBS uniform bins over [-WINDOW_DAYS, WINDOW_DAYS], i.e.
# `--window-mode fixed --window-days 0.2 --n-bins 50`. The default builder mode
# is `auto`, which gives every target a different grid and breaks the NPE.
#
# Usage:  ./run_pipeline.sh [stage ...]
#   stages: build  train  apply  pit   (default: build train apply; pit is opt-in)
#   e.g.  ./run_pipeline.sh train apply    # skip the library build
#         ./run_pipeline.sh train          # retrain only
#         ./run_pipeline.sh pit            # PIT calibration plot only
#   COMPRESSION=hybrid ./run_pipeline.sh train
#   MAX_TARGETS=50 ./run_pipeline.sh       # env var, only affects `build`
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}

MAX_TARGETS=${MAX_TARGETS:-50}
COMPRESSION=${COMPRESSION:-weighted}

case "${1:-}" in
    -h|--help)
        cat <<'EOF'
Usage: ./run_pipeline.sh [stage ...]

Run the transit-SBI pipeline. With no stages, runs build train apply in order.
(pit is opt-in and never runs by default.)

Stages:
  build   Build the DR25 DV library on the fixed 50-bin grid
  train   Train the noise-aware weighted/hybrid NPE (-> weights/)
  apply   Compare NPE vs MCMC on mocks and Kepler targets (-> plots/)
  pit     PIT calibration check of the trained NPE   (-> plots/pit_<mode>.png)

Examples:
  ./run_pipeline.sh                 # full pipeline (build train apply)
  ./run_pipeline.sh train apply     # skip the library build
  ./run_pipeline.sh train           # retrain only
  ./run_pipeline.sh pit             # calibration plot only

Env:
  MAX_TARGETS   number of targets for `build` (default 50)
  COMPRESSION   weighted or hybrid (default weighted)
  N_KEPLER      held-out targets spanning S/N for `apply` (default 1)
EOF
        exit 0
        ;;
esac

# Stages to run: named args, or build/train/apply if none given (pit is opt-in).
STAGES="${*:-build train apply}"
for s in $STAGES; do
    case "$s" in
        build|train|apply|pit) ;;
        *) echo "Unknown stage '$s' (valid: build train apply pit)" >&2; exit 1 ;;
    esac
done
if [[ ! -x "$PY" ]]; then
    echo "Python not found at '$PY'. Create .venv as shown in README.md, or set PY." >&2
    exit 1
fi
run() { [[ " $STAGES " == *" $1 "* ]]; }

if run build; then
    echo "=== Building DR25 DV library (fixed grid: 50 bins over +/-0.2 d) ==="
    "$PY" build_dr25_dv_library.py \
        --window-mode fixed \
        --window-days 0.2 \
        --n-bins 50 \
        --min-snr 7.1 \
        --max-depth-ppm 50000 \
        --max-targets "$MAX_TARGETS"
fi

if run train; then
    echo "=== Training noise-aware $COMPRESSION NPE ==="
    "$PY" train_sbi.py --compression "$COMPRESSION"
fi

if run apply; then
    echo "=== Comparing NPE vs MCMC on mocks and Kepler targets ==="
    "$PY" example_transit.py
fi

if run pit; then
    echo "=== PIT calibration check (-> plots/pit_<mode>.png) ==="
    "$PY" pit_plot.py
fi

echo "=== Done. Model in weights/, results in plots/. ==="
