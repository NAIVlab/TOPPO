#!/bin/bash
# MT50 baseline grid launcher: submits one Slurm job per (arm, seed) using the
# sbatch script shipped in this repo.
#
#   scripts/launch_mt50.sh <arm> <seed> [<seed> ...]     # e.g. ars 0 1 2
#   ARMS: ars | ars_ln | ars_ln1024 | ars_lora | mtsac | mtsac_cb | mtsac_fullstack
#
# Partition / QOS / account are site-specific and are NOT set here; pass them
# through SBATCH_EXTRA, e.g.
#
#   SBATCH_EXTRA="--partition=gpu --qos=normal" scripts/launch_mt50.sh ars 0 1 2
#
# Job shape is calibrated for one RTX 3090: ~57 it/s steady, ~14 h per run,
# MaxRSS ~48 G at production shape, so 64 G leaves headroom. An OOM is
# recoverable by resubmitting with more memory — the run resumes from its
# checkpoint.
#
# wandb: name sac_mt50_<arm>_s<seed>, group = name minus _s<seed>, tags mt50,sac.
# The arm table lives in slurm/site_3090_mt50_run.sbatch; this script only
# fans out over seeds so the two cannot drift apart.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
SBATCH_SCRIPT="${PROJ}/slurm/site_3090_mt50_run.sbatch"

if [ $# -lt 2 ]; then
    sed -n '2,20p' "${BASH_SOURCE[0]}"
    exit 2
fi
ARM=$1; shift
mkdir -p "${PROJ}/logs"   # the sbatch writes logs/%x_%j.out under --chdir

for S in "$@"; do
    NAME="sac_mt50_${ARM}_s${S}"
    echo ">>> ${NAME}"
    # shellcheck disable=SC2086
    sbatch --job-name="${NAME}" --chdir="${PROJ}" ${SBATCH_EXTRA:-} \
        "${SBATCH_SCRIPT}" "${ARM}" "${S}"
done
