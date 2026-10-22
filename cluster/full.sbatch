#!/usr/bin/env bash
#SBATCH --job-name=elicitor-full
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.log
# The full run (SPEC §1, §13, §15): one job, at most 12 h on 4 GPUs. A job that hits the limit
# is resubmitted with the same command and resumes. Submit from the repo root:
#   cluster/submit.sh cluster/full.sbatch [experiments/m5_full.toml]
set -eo pipefail
export CONFIG="${1:-experiments/m5_full.toml}"
export REQUIRE_GPU=1
source cluster/job_setup.sh
source cluster/serve_models.sh
trap stop_models EXIT
name=$(basename "$CONFIG" .toml)
sha256sum data/travel/*.jsonl

start=$(date +%s)
elicitor run "$CONFIG" --workers 32 --out "results/$name"
echo "full run: $(( $(date +%s) - start )) s"
elicitor health "results/$name"
elicitor analyze "results/$name" --publish
