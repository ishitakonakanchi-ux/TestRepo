#!/bin/bash
# Build the Kepler DR25 library, train a noise-aware amortized NPE, and
# compare it with NUTS on two mocks and one or more Kepler targets.
#
# The library MUST be binned on the same fixed grid the simulator uses (t_obs in
# transit_sbi.py): N_OBS uniform bins over [-WINDOW_DAYS, WINDOW_DAYS], i.e.
# `--window-mode fixed --window-days 0.2 --n-bins 50`. The wrapper pins these
# values explicitly so future builder-default changes cannot alter the NPE grid.
#
# Usage:  ./run_pipeline.sh [stage ...]
#   stages: build train apply refine pit check
#   default: build train refine pit
#   e.g.  ./run_pipeline.sh train apply    # skip the library build
#         ./run_pipeline.sh refine         # guarded NPE-IS accuracy mode
#         ./run_pipeline.sh pit            # PIT calibration plot only
#         ./run_pipeline.sh check          # fast executable checks
#   COMPRESSION=core_domain ./run_pipeline.sh train pit
#   MAX_TARGETS=50 ./run_pipeline.sh       # env var, only affects `build`
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}

MAX_TARGETS=${MAX_TARGETS:-50}
COMPRESSION=${COMPRESSION:-robust}
WEIGHTS=${WEIGHTS:-}
PIT_NOISE_MODEL=${PIT_NOISE_MODEL:-native}
N_KEPLER=${N_KEPLER:-4}
NPE_IMPORTANCE_REFINE=${NPE_IMPORTANCE_REFINE:-0}
export COMPRESSION N_KEPLER

case "${1:-}" in
    -h|--help)
        cat <<'EOF'
Usage: ./run_pipeline.sh [stage ...]

Run the transit-SBI pipeline. With no stages, runs build train refine pit in order.

Stages:
  build    Build the DR25 DV library on the fixed 50-bin grid
  train    Train the selected NPE mode
  apply    Compare the selected NPE with NUTS on mocks and Kepler targets
  refine   Run guarded NPE-IS accuracy refinement (full 7-parameter model only)
  pit      Run native or cross-noise PIT calibration
  check    Run compression tests, Python compilation, and shell syntax checks

Examples:
  ./run_pipeline.sh                 # full pipeline (build train refine pit)
  ./run_pipeline.sh train apply     # skip the library build
  COMPRESSION=core ./run_pipeline.sh train apply pit
  WEIGHTS=weights/npe_robust_<timestamp>.pkl ./run_pipeline.sh refine
  COMPRESSION=core_domain PIT_NOISE_MODEL=white ./run_pipeline.sh pit
  ./run_pipeline.sh check

Env:
  MAX_TARGETS   number of targets for `build` (default 50)
  COMPRESSION   weighted, hybrid, full, embedded, robust, core, or core_domain
                (default robust)
  WEIGHTS       exact model path for apply/refine/pit; otherwise the newest
                compatible COMPRESSION model is selected
  N_KEPLER      held-out targets spanning S/N for `apply`/`refine` (default 4)
  PIT_NOISE_MODEL native, white, or domain for `pit` (default native)
  NPE_DEVICE    torch device for `train`: cpu, mps, or cuda (default cpu)
  NPE_CPU_THREADS CPU helper threads with CUDA: 1 or 2 (default 2)
  NPE_IMPORTANCE_REFINE=1 also enables guarded PSIS likelihood correction
                during `apply`; `refine` sets it automatically
  NPE_POOL_SIZE, NPE_REFRESH_EVERY, NPE_SIMS_PER_EPOCH, NPE_EPOCHS, and
                NPE_PATIENCE tune `train` (see README.md)
EOF
        exit 0
        ;;
esac

# Stages to run: named args, or build/train/refine/pit if none were supplied.
if (( $# == 0 )); then
    STAGES=(build train refine pit)
else
    STAGES=("$@")
fi
for s in "${STAGES[@]}"; do
    case "$s" in
        build|train|apply|refine|pit|check) ;;
        *)
            echo "Unknown stage '$s'" >&2
            echo "Valid stages: build train apply refine pit check" >&2
            exit 1
            ;;
    esac
done
if [[ ! -x "$PY" ]]; then
    echo "Python not found at '$PY'. Create .venv as shown in README.md, or set PY." >&2
    exit 1
fi
run() {
    local wanted=$1 stage
    for stage in "${STAGES[@]}"; do
        [[ "$stage" == "$wanted" ]] && return 0
    done
    return 1
}

case "$COMPRESSION" in
    weighted|hybrid|full|embedded|robust|core|core_domain) ;;
    *)
        echo "Invalid COMPRESSION '$COMPRESSION'." >&2
        echo "Use weighted, hybrid, full, embedded, robust, core, or core_domain." >&2
        exit 1
        ;;
esac

MODEL_ARGS=()
if (run apply || run refine || run pit) && [[ -n "$WEIGHTS" ]]; then
    if [[ ! -f "$WEIGHTS" ]]; then
        echo "Weights file not found: $WEIGHTS" >&2
        exit 1
    fi
    MODEL_ARGS=("$WEIGHTS")
fi

if (run apply || run refine) && [[ ! "$N_KEPLER" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_KEPLER must be a positive integer, got '$N_KEPLER'." >&2
    exit 1
fi
if (run apply || run refine) \
        && [[ "$NPE_IMPORTANCE_REFINE" != 0 \
              && "$NPE_IMPORTANCE_REFINE" != 1 ]]; then
    echo "NPE_IMPORTANCE_REFINE must be 0 or 1." >&2
    exit 1
fi
if run refine && [[ -z "$WEIGHTS" && "$COMPRESSION" == core* ]]; then
    echo "The refine stage requires a full seven-parameter model." >&2
    echo "Use COMPRESSION=robust or set WEIGHTS to full-model weights." >&2
    exit 1
fi

if run build; then
    if [[ ! "$MAX_TARGETS" =~ ^[1-9][0-9]*$ ]]; then
        echo "MAX_TARGETS must be a positive integer, got '$MAX_TARGETS'." >&2
        exit 1
    fi
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
    if [[ "${NPE_DEVICE:-cpu}" == cuda* ]]; then
        NPE_CPU_THREADS=${NPE_CPU_THREADS:-2}
        if [[ ! "$NPE_CPU_THREADS" =~ ^[12]$ ]]; then
            echo "NPE_CPU_THREADS must be 1 or 2 for CUDA training." >&2
            exit 1
        fi
        export NPE_CPU_THREADS
        export OMP_NUM_THREADS="$NPE_CPU_THREADS"
        export MKL_NUM_THREADS="$NPE_CPU_THREADS"
        export OPENBLAS_NUM_THREADS="$NPE_CPU_THREADS"
        export NUMEXPR_NUM_THREADS="$NPE_CPU_THREADS"
        echo "=== Limiting CUDA training to $NPE_CPU_THREADS CPU threads ==="
    fi
    echo "=== Training noise-aware $COMPRESSION NPE ==="
    "$PY" train_sbi.py --compression "$COMPRESSION"
fi

if run apply; then
    echo "=== Comparing NPE vs NUTS on mocks and Kepler targets ==="
    "$PY" example_transit.py "${MODEL_ARGS[@]}"
fi

if run refine; then
    echo "=== Guarded NPE-IS refinement vs NUTS ==="
    NPE_IMPORTANCE_REFINE=1 "$PY" example_transit.py "${MODEL_ARGS[@]}"
fi

if run pit; then
    case "$PIT_NOISE_MODEL" in
        native|white|domain) ;;
        *)
            echo "PIT_NOISE_MODEL must be native, white, or domain." >&2
            exit 1
            ;;
    esac
    echo "=== PIT calibration ($PIT_NOISE_MODEL noise) ==="
    PIT_NOISE_MODEL="$PIT_NOISE_MODEL" \
        "$PY" pit_plot.py "${MODEL_ARGS[@]}"
fi

if run check; then
    echo "=== Running lightweight validation checks ==="
    "$PY" test_noise_aware.py
    "$PY" -m py_compile ./*.py
    bash -n run_pipeline.sh
fi

echo "=== Done. Outputs, when produced, are in weights/ and plots/. ==="
