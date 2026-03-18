#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT_DIR/rag-backend"
FRONTEND_DIR="$ROOT_DIR/rag-frontend"
BACKEND_PORT=8000
FRONTEND_PORT=5173
BACKEND_LOG="$ROOT_DIR/backend.dev.log"
FRONTEND_LOG="$ROOT_DIR/frontend.dev.log"

kill_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"$port" || true)"
  if [[ -n "$pids" ]]; then
    while IFS= read -r pid; do
      [[ -n "$pid" ]] && kill -9 "$pid"
    done <<< "$pids"
  fi
}

kill_port "$BACKEND_PORT"
kill_port "$FRONTEND_PORT"

cd "$BACKEND_DIR"
nohup uv run python main.py > "$BACKEND_LOG" 2>&1 &

cd "$FRONTEND_DIR"
nohup npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" > "$FRONTEND_LOG" 2>&1 &

sleep 1

echo "backend: http://localhost:$BACKEND_PORT"
echo "frontend: http://localhost:$FRONTEND_PORT"
echo "backend log: $BACKEND_LOG"
echo "frontend log: $FRONTEND_LOG"
