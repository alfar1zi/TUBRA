#!/usr/bin/env bash
# Measure API latency (10 sequential requests)
# Usage: ./scripts/measure_latency.sh <BASE_URL>

set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"
ENDPOINT="/api/v1/health"
REQUESTS=10

echo "Measuring latency: $BASE_URL$ENDPOINT ($REQUESTS requests)"
echo ""

total=0
for i in $(seq 1 $REQUESTS); do
  start=$(date +%s%3N)
  if curl -sf "$BASE_URL$ENDPOINT" >/dev/null; then
    end=$(date +%s%3N)
    latency=$((end - start))
    total=$((total + latency))
    printf "Request %2d: %4d ms\n" "$i" "$latency"
  else
    printf "Request %2d: FAILED\n" "$i"
  fi
  sleep 0.5
done

if [[ $total -gt 0 ]]; then
  avg=$((total / REQUESTS))
  echo ""
  echo "Average: ${avg} ms"
fi