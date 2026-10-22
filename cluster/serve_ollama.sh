#!/usr/bin/env bash
# Start an Ollama server on a free local port and wait until it answers.
# Source it from a job script:  source cluster/serve_ollama.sh
# Afterwards ELICITOR_BASE_URL points at the server and OLLAMA_PID holds its process id.
# A shared read-only model folder needs OLLAMA_NOPRUNE=1, or the server tries to clean it up.

port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export OLLAMA_HOST="127.0.0.1:$port"
export OLLAMA_NOPRUNE=1
export ELICITOR_BASE_URL="http://$OLLAMA_HOST"
ollama serve > "${OLLAMA_LOG:-ollama-$port.log}" 2>&1 &
OLLAMA_PID=$!
for _ in $(seq 1 120); do
    if curl -sf "$ELICITOR_BASE_URL/api/tags" > /dev/null; then
        echo "ollama serving at $ELICITOR_BASE_URL (pid $OLLAMA_PID)"
        return 0 2>/dev/null || exit 0
    fi
    sleep 1
done
echo "ollama did not start within 120 s" >&2
kill "$OLLAMA_PID" 2>/dev/null
return 1 2>/dev/null || exit 1
