#!/usr/bin/env bash
set -euo pipefail

WHEEL_PATH="${1:?wheel path is required}"
RELEASE_ID="${2:-$(date +%Y%m%d%H%M%S)}"
BASE_DIR="${TIMECAMPUS_AGENT_HOME:-$HOME/timecampus-agent}"
RELEASE_DIR="$BASE_DIR/releases/$RELEASE_ID"
CURRENT_LINK="$BASE_DIR/current"
SERVICE_NAME="${TIMECAMPUS_AGENT_SERVICE:-timecampus-agent}"
HEALTH_URL="${TIMECAMPUS_AGENT_HEALTH_URL:-http://127.0.0.1:8090/health}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREVIOUS_TARGET="$(readlink -f "$CURRENT_LINK" 2>/dev/null || true)"

mkdir -p "$RELEASE_DIR" "$BASE_DIR/releases" "$BASE_DIR/shared/memory" \
  "$BASE_DIR/shared/eval-reports"
cp "$WHEEL_PATH" "$RELEASE_DIR/"

uv venv --python 3.12 "$RELEASE_DIR/.venv"
uv pip install --python "$RELEASE_DIR/.venv/bin/python" "$RELEASE_DIR/"*.whl

sudo install -m 0644 "$SCRIPT_DIR/timecampus-agent.service" \
  "/etc/systemd/system/$SERVICE_NAME.service"
ln -sfn "$RELEASE_DIR" "$CURRENT_LINK.next"
mv -Tf "$CURRENT_LINK.next" "$CURRENT_LINK"

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

for _ in $(seq 1 30); do
  if curl -fsS "$HEALTH_URL" >/dev/null; then
    find "$BASE_DIR/releases" -mindepth 1 -maxdepth 1 -type d \
      -printf '%T@ %p\n' | sort -nr | tail -n +6 | cut -d' ' -f2- | xargs -r rm -rf
    echo "Agent release $RELEASE_ID is healthy"
    exit 0
  fi
  sleep 1
done

if [ -n "$PREVIOUS_TARGET" ] && [ -d "$PREVIOUS_TARGET" ]; then
  ln -sfn "$PREVIOUS_TARGET" "$CURRENT_LINK.next"
  mv -Tf "$CURRENT_LINK.next" "$CURRENT_LINK"
  sudo systemctl restart "$SERVICE_NAME"
fi
echo "Agent health check failed; previous release restored" >&2
exit 1
