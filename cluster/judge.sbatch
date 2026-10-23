#!/usr/bin/env bash
#SBATCH --job-name=elicitor-judge
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log
# The M7 judge study (SPEC Q10, §1, §17): LLM judges score a finished run's recommendations,
# with no new dialogue. At most 3 h on 4 GPUs (two servers per model); the smoke file on the
# M5 smoke run stays under 1 h. A rerun is free through the judge folder's reply cache.
# Submit from the repo root, where the source run folder is:
#   cluster/submit.sh cluster/judge.sbatch [experiments/m7_judge.toml]
set -eo pipefail
export CONFIG="${1:-experiments/m7_judge.toml}"
export REQUIRE_GPU=1
source cluster/job_setup.sh
source cluster/serve_models.sh
trap stop_models EXIT
name=$(basename "$CONFIG" .toml)
sha256sum data/travel/catalog.jsonl data/travel/personas.jsonl data/travel/prompts/*.txt

start=$(date +%s)
status=0
elicitor judge "$CONFIG" --workers 32 --out "results/$name" || status=$?
echo "judge: $(( $(date +%s) - start )) s (exit $status)"
if [[ "$name" == *smoke* ]]; then
    elicitor judge-report "results/$name"
else
    elicitor judge-report "results/$name" --publish
fi
exit "$status"
