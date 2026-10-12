#!/usr/bin/env bash
# Local development runner for TBCare
# Usage: ./scripts/run_local.sh [up|down|logs|restart]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$ROOT_DIR/docker-compose.yml"

cmd="${1:-up}"

case "$cmd" in
  up)
    echo "Building and starting services..."
    docker compose -f "$COMPOSE_FILE" up --build -d
    echo "Waiting for API health..."
    for i in {1..30}; do
      if curl -sf http://localhost:8080/api/v1/health >/dev/null 2>&1; then
        echo "API ready at http://localhost:8080"
        break
      fi
      sleep 2
    done
    echo "Web available at http://localhost:5173"
    echo "API available at http://localhost:8080"
    echo "Demo mode: http://localhost:5173?demo=1"
    ;;
  down)
    echo "Stopping services..."
    docker compose -f "$COMPOSE_FILE" down
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" logs -f "${2:-}"
    ;;
  restart)
    docker compose -f "$COMPOSE_FILE" restart "${2:-}"
    ;;
  *)
    echo "Usage: $0 {up|down|logs|restart} [service]"
    exit 1
    ;;
esac