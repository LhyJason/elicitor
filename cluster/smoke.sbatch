#!/usr/bin/env bash
#SBATCH --job-name=elicitor-smoke
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
# One-GPU smoke run (SPEC §15, §17 M2). Submit from the repo root:
#   cluster/submit.sh cluster/smoke.sbatch [experiments/m2_smoke_cluster.toml]
# Expects a venv in .venv with the package installed (pip install -e .).
# No `set -u`: module scripts read unset variables.
set -eo pipefail
config="${1:-experiments/m2_smoke_cluster.toml}"

source cluster/local.env
if [[ -n "${PYTHON_MODULE:-}" ]]; then module load "$PYTHON_MODULE"; fi
if [[ -n "${OLLAMA_MODULE:-}" ]]; then module load "$OLLAMA_MODULE"; fi
# The ollama module sets its own OLLAMA_MODELS; the site setting wins.
source cluster/local.env
export OLLAMA_MODELS
source .venv/bin/activate
export PYTHONUNBUFFERED=1  # progress lines reach the log as they happen

nvidia-smi --query-gpu=name,memory.total --format=csv
export OLLAMA_LOG="logs/ollama-$SLURM_JOB_ID.log"
source cluster/serve_ollama.sh
trap 'kill "$OLLAMA_PID" 2>/dev/null' EXIT

start=$(date +%s)
elicitor run "$config" --out "results/$(basename "$config" .toml)"
echo "wall time: $(( $(date +%s) - start )) s"
