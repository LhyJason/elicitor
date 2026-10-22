#!/usr/bin/env bash
# Start one Ollama server per GPU and load each group's model (SPEC §15, M5).
# Source it from a job script after activating the venv:
#   CONFIG=experiments/m5_full.toml source cluster/serve_models.sh
# Settings (environment): AGENT_GPUS and USER_GPUS (servers per model group, default 2 each),
# AGENT_PARALLEL and USER_PARALLEL (OLLAMA_NUM_PARALLEL, default 8 and 4), REQUIRE_GPU=1 to stop
# when a model does not sit fully on its GPU. With AGENT_GPUS=0 both groups share one server.
# Afterwards ELICITOR_SERVERS_AGENT / _USER list the servers, OLLAMA_PIDS their process ids and
# stop_models stops them.

AGENT_GPUS=${AGENT_GPUS:-2}
USER_GPUS=${USER_GPUS:-2}
AGENT_PARALLEL=${AGENT_PARALLEL:-8}
USER_PARALLEL=${USER_PARALLEL:-4}
read -r AGENT_MODEL USER_MODEL NUM_CTX < <(python - "$CONFIG" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    llm = tomllib.load(f)["llm"]
print(llm["models"]["agent"], llm["models"]["user"], llm.get("num_ctx", 4096))
PY
)
export OLLAMA_NOPRUNE=1  # the shared model folder is read-only
OLLAMA_PIDS=()

# start_server <gpu index> <num_parallel> <max loaded models>: sets SERVER_URL.
start_server() {
    local port
    port=$(python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    CUDA_VISIBLE_DEVICES=$1 OLLAMA_HOST="127.0.0.1:$port" OLLAMA_NUM_PARALLEL=$2 \
        OLLAMA_MAX_LOADED_MODELS=$3 OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1 \
        OLLAMA_CONTEXT_LENGTH="$NUM_CTX" \
        ollama serve > "logs/ollama-${SLURM_JOB_ID:-local}-gpu$1.log" 2>&1 &
    OLLAMA_PIDS+=($!)
    SERVER_URL="http://127.0.0.1:$port"
    for _ in $(seq 1 120); do
        if curl -sf "$SERVER_URL/api/tags" > /dev/null; then
            echo "gpu $1: ollama at $SERVER_URL (parallel $2)"
            return 0
        fi
        sleep 1
    done
    echo "gpu $1: ollama did not start within 120 s" >&2
    return 1
}

# load_model <url> <model>: blocks until the model is in memory.
load_model() {
    curl -sf "$1/api/generate" -o /dev/null \
        -d "{\"model\": \"$2\", \"keep_alive\": -1, \"options\": {\"num_ctx\": $NUM_CTX}}"
}

stop_models() {
    kill "${OLLAMA_PIDS[@]}" 2>/dev/null
}

agent_urls=()
user_urls=()
if [[ "$AGENT_GPUS" == 0 ]]; then
    start_server 0 "$AGENT_PARALLEL" 2 || return 1
    agent_urls+=("$SERVER_URL")
    user_urls+=("$SERVER_URL")
else
    for ((i = 0; i < AGENT_GPUS; i++)); do
        start_server "$i" "$AGENT_PARALLEL" 1 || return 1
        agent_urls+=("$SERVER_URL")
    done
    for ((i = AGENT_GPUS; i < AGENT_GPUS + USER_GPUS; i++)); do
        start_server "$i" "$USER_PARALLEL" 1 || return 1
        user_urls+=("$SERVER_URL")
    done
fi

start=$(date +%s)
load_pids=()
for url in "${agent_urls[@]}"; do load_model "$url" "$AGENT_MODEL" & load_pids+=($!); done
for url in "${user_urls[@]}"; do load_model "$url" "$USER_MODEL" & load_pids+=($!); done
# Wait for the loads only: a bare `wait` would also wait for the servers.
for pid in "${load_pids[@]}"; do
    if ! wait "$pid"; then echo "a model failed to load" >&2; stop_models; return 1; fi
done
echo "models loaded in $(( $(date +%s) - start )) s (num_ctx $NUM_CTX)"

gpu_ok=1
for url in $(printf '%s\n' "${agent_urls[@]}" "${user_urls[@]}" | sort -u); do
    ps_out=$(OLLAMA_HOST="${url#http://}" ollama ps)
    echo "$url"; echo "$ps_out"
    if echo "$ps_out" | tail -n +2 | grep -qv "100% GPU"; then gpu_ok=0; fi
done
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
if [[ "$gpu_ok" == 0 ]]; then
    echo "WARNING: a model is not fully on its GPU (see ollama ps above)" >&2
    if [[ "${REQUIRE_GPU:-0}" == 1 ]]; then stop_models; return 1; fi
fi

ELICITOR_SERVERS_AGENT=$(IFS=,; echo "${agent_urls[*]}")
ELICITOR_SERVERS_USER=$(IFS=,; echo "${user_urls[*]}")
export ELICITOR_SERVERS_AGENT ELICITOR_SERVERS_USER
