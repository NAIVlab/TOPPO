#!/usr/bin/env bash
# Short wall-clock measurement of one configuration (the H200 timing sentence in
# the paper: ~8M samples, seed 260511470, training command only; the separate
# --mode test evaluation of run_arm.sh is not run).
#
#   bash scripts/time_arm.sh <ASSEMBLED_TREE> <wandb_group> [max_samples]
#
# Prints the wall-clock seconds of the training command with --max_samples
# (default 8000000) and output/timing_<group>/ as the output directory. Compare
# configurations only on the same GPU type, CPU count and memory: environment
# stepping is CPU-bound enough that the CPU count alone changes throughput.
# results/timing/timing.tsv lists the published measurements (H200 NVL, 8 CPU cores).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE="$(cd "${1:?usage: time_arm.sh <tree> <group> [max_samples]}" && pwd)"
GROUP="${2:?usage: time_arm.sh <tree> <group> [max_samples]}"
MAX_SAMPLES="${3:-8000000}"
PYTHON="${PYTHON:-python}"

CMD="$("$PYTHON" "$HERE/campaign.py" train "$GROUP" 260511470 --python "$PYTHON" \
        --max-samples "$MAX_SAMPLES" --out-dir "output/timing_${GROUP}/")"
cd "$TREE"
echo "$CMD"
T0="$(date +%s.%N)"
eval "$CMD"
T1="$(date +%s.%N)"
"$PYTHON" -c 'import sys; print("wall_s\t%.1f" % (float(sys.argv[2]) - float(sys.argv[1])))' "$T0" "$T1"
