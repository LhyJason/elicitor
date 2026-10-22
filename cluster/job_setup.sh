#!/usr/bin/env bash
# Common setup of the job scripts: site settings, modules, venv. Source it from the repo root.
# No `set -u`: module scripts read unset variables.
source cluster/local.env
if [[ -n "${PYTHON_MODULE:-}" ]]; then module load "$PYTHON_MODULE"; fi
if [[ -n "${OLLAMA_MODULE:-}" ]]; then module load "$OLLAMA_MODULE"; fi
# The ollama module sets its own OLLAMA_MODELS; the site setting wins.
source cluster/local.env
export OLLAMA_MODELS
source .venv/bin/activate
export PYTHONUNBUFFERED=1  # progress lines reach the log as they happen
echo "code $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (modified)') on $(hostname)"
