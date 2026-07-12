#!/bin/bash
# Build the Kepler DR25 library, train the amortized MLE-compression NPE, and
# apply it to every eligible target versus MCMC.
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
#   MAX_TARGETS=30 ./run_pipeline.sh       # env var, only affects `build`
set -euo pipefail

# Point this at your own virtualenv if different. It must have:
#   sbi jax jaxoplanet numpyro corner torch numpy matplotlib astropy tqdm
PY=/Users/rstiskalek/Projects/Teaching/venv_teach/bin/python
cd "$(dirname "$0")"

MAX_TARGETS=${MAX_TARGETS:-20}

case "${1:-}" in
    -h|--help)
        cat <<'EOF'
Usage: ./run_pipeline.sh [stage ...]

Run the transit-SBI pipeline. With no stages, runs build train apply in order.
(pit is opt-in and never runs by default.)

Stages:
  build   Build the DR25 DV library on the fixed 50-bin grid
  train   Train the amortized MLE-compression NPE   (-> weights/)
  apply   Apply NPE vs MCMC across targets           (-> plots/)
  pit     PIT calibration check of the trained NPE   (-> plots/pit.png)

Examples:
  ./run_pipeline.sh                 # full pipeline (build train apply)
  ./run_pipeline.sh train apply     # skip the library build
  ./run_pipeline.sh train           # retrain only
  ./run_pipeline.sh pit             # calibration plot only

Env:
  MAX_TARGETS   number of targets for `build` (default 20)
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
run() { [[ " $STAGES " == *" $1 "* ]]; }

if run build; then
    echo "=== Building DR25 DV library (fixed grid: 50 bins over +/-0.2 d) ==="
    "$PY" build_dr25_dv_library.py \
        --window-mode fixed \
        --window-days 0.2 \
        --n-bins 50 \
        --max-targets "$MAX_TARGETS"
fi

if run train; then
    echo "=== Training amortized MLE-compression NPE ==="
    "$PY" train_sbi.py
fi

if run apply; then
    echo "=== Applying NPE vs MCMC across eligible targets ==="
    "$PY" example_transit.py
fi

if run pit; then
    echo "=== PIT calibration check (-> plots/pit.png) ==="
    "$PY" pit_plot.py
fi

echo "=== Done. Model in weights/, results in plots/. ==="
