#!/usr/bin/env bash
# Fan out the MT50 (default) or MT10 baseline grid on an A100 Slurm site;
# BENCHMARK=MT10 selects the MT10 reruns (5 arms, see mt50_run.sbatch).
# DRY-RUN by default — prints every sbatch command; add -y to actually submit.
#
# Usage (from the ARS repo root):
#   bash slurm/site_a100/launch_mt50_grid.sh [-y] ["arms"] ["seeds"]
# Defaults (the 7 SAC-family arms of the paper, 10 seeds each):
#   arms  = "ars ars_ln ars_ln1024 ars_lora mtsac mtsac_cb mtsac_fullstack"
#   seeds = "0 1 2 3 4 5 6 7 8 9"
# BENCHMARK=MT10: the arms default to the five MT10 arms (50 jobs); the two
# transplant arms (mtsac_cb, mtsac_fullstack) are MT50 only and are refused.
# Examples:
#   bash slurm/site_a100/launch_mt50_grid.sh                          # print all 70
#   bash slurm/site_a100/launch_mt50_grid.sh -y "ars_ln1024" "5 6 7"  # submit 3
#
# !! Jobs with the same (arm, seed) share a wandb name and a checkpoint dir — if
# another wave of the same grid is running, use disjoint seeds or
# `export NAME_SUFFIX=_wave2` first (see mt50_run.sbatch).
set -uo pipefail
submit=0; pos=()
for arg in "$@"; do
  if [ "$arg" = "-y" ]; then submit=1; else pos+=("$arg"); fi
done
if [ "${BENCHMARK:-MT50}" = MT10 ]; then default_arms="ars ars_ln ars_ln1024 ars_lora mtsac"
else default_arms="ars ars_ln ars_ln1024 ars_lora mtsac mtsac_cb mtsac_fullstack"; fi
arms="${pos[0]:-$default_arms}"
bm=$(printf '%s' "${BENCHMARK:-MT50}" | tr '[:upper:]' '[:lower:]')
seeds="${pos[1]:-0 1 2 3 4 5 6 7 8 9}"
for a in $arms; do
  case "$a" in ars|ars_ln|ars_ln1024|ars_lora|mtsac|mtsac_cb|mtsac_fullstack) ;; *) echo "bad arm: '$a'" >&2; exit 2 ;; esac
done
ARMS_CHECK=$arms
case "${BENCHMARK:-MT50}" in
  MT50) ;;
  MT10) case " $ARMS_CHECK " in *" mtsac_cb "*|*" mtsac_fullstack "*) echo "mtsac_cb / mtsac_fullstack were run on MT50 only" >&2; exit 2 ;; esac ;;
  *) echo "bad BENCHMARK: '${BENCHMARK}' (MT50 | MT10)" >&2; exit 2 ;;
esac
for s in $seeds; do
  case "$s" in ''|*[!0-9]*) echo "bad seed: '$s'" >&2; exit 2 ;; esac
done
mkdir -p logs
n=0
run() { n=$((n+1)); if [ "$submit" = 1 ]; then "$@"; else echo "DRY: $*"; fi; }
for a in $arms; do
  for s in $seeds; do
    run sbatch --job-name="sac_${bm}_${a}_s${s}${NAME_SUFFIX:-}" \
        slurm/site_a100/mt50_run.sbatch "$a" "$s"
  done
done
if [ "$submit" = 1 ]; then
  echo "[launch_grid] submitted $n jobs (~17-24 h each on A100; ars_ln1024 ~28-36 h)"
  echo "NOTE: WANDB_MODE=${WANDB_MODE:-online}; only offline runs need a periodic"
  echo "      wandb sync --sync-all wandb/   from a login node (--append for resumed runs)"
else
  echo "[launch_grid] DRY-RUN: would submit $n jobs (add -y to submit)"
fi
