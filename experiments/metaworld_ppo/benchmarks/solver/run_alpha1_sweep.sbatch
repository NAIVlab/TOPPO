#!/bin/bash
# Synthetic alpha=1 spike-scenario sweep for the FairGrad y-Newton solver.
#
# Partition / QOS / account are site-specific -- pass them on the command line.
# Run from the experiments/metaworld_ppo/ directory (or pass --chdir), and create
# run_logs/ first: Slurm opens the --output file before the script starts and
# does not create missing directories, so the job fails without a log otherwise.
#   mkdir -p run_logs
#   sbatch --partition=<gpu-partition> [--qos=<qos>] [--account=<acct>] \
#          benchmarks/solver/run_alpha1_sweep.sbatch
#SBATCH --job-name=solver_a1
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=run_logs/solver_a1_%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 SPIKE_ALPHA=1.0
echo "node: $(hostname)  cpu: $(lscpu | grep 'Model name' | head -1)"
time python benchmarks/solver/fairgrad_spike_scenarios_a1.py
