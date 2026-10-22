#!/usr/bin/env bash
# Submit a job script with the account and partition from cluster/local.env.
#   cluster/submit.sh cluster/smoke.sbatch [experiment.toml]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -f "$here/local.env" ]]; then
    echo "missing $here/local.env (copy local.env.example and fill it in)" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$here/local.env"
mkdir -p "$here/../logs"
exec sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" "$@"
