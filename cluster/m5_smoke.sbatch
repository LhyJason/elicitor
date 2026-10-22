#!/usr/bin/env bash
#SBATCH --job-name=elicitor-m5-smoke
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
# M5 smoke job (SPEC §15, §17): backstories (only if not written yet), rho calibration on the
# dev personas, then the smoke run with health. Submit from the repo root:
#   cluster/submit.sh cluster/m5_smoke.sbatch [experiments/m5_smoke.toml]
set -eo pipefail
export CONFIG="${1:-experiments/m5_smoke.toml}"
source cluster/job_setup.sh
source cluster/serve_models.sh
trap stop_models EXIT
name=$(basename "$CONFIG" .toml)

if [[ ! -f data/travel/backstories.jsonl ]]; then
    start=$(date +%s)
    elicitor make-backstories "$CONFIG" --workers 16 || echo "some backstories stay flagged"
    echo "backstories: $(( $(date +%s) - start )) s"
fi
sha256sum data/travel/backstories.jsonl

start=$(date +%s)
elicitor calibrate "$CONFIG" --workers 32 --out "results/${name}_calibration"
echo "calibration: $(( $(date +%s) - start )) s"

start=$(date +%s)
elicitor run "$CONFIG" --workers 32 --out "results/$name"
echo "smoke run: $(( $(date +%s) - start )) s"
elicitor health "results/$name" || true
