#!/usr/bin/env bash
#SBATCH --job-name=elicitor-smoke
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
# One-GPU smoke run (SPEC §15, §17 M2): both model groups on one server. Submit from the repo
# root:
#   cluster/submit.sh cluster/smoke.sbatch [experiments/m2_smoke_cluster.toml]
# Expects a venv in .venv with the package installed (pip install -e .).
set -eo pipefail
export CONFIG="${1:-experiments/m2_smoke_cluster.toml}"
export AGENT_GPUS=0 AGENT_PARALLEL=1
source cluster/job_setup.sh
source cluster/serve_models.sh
trap stop_models EXIT

start=$(date +%s)
elicitor run "$CONFIG" --out "results/$(basename "$CONFIG" .toml)"
echo "wall time: $(( $(date +%s) - start )) s"
