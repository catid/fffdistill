#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-all}"
REMOTE_DIR="${2:-/home/catid/fffdistill}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if ! git -C "$REPO_ROOT" diff --quiet; then
  echo "Refusing to sync uncommitted local worktree changes. Commit first." >&2
  exit 2
fi

sync_one() {
  local host="$1"
  echo "==> syncing ${host}:${REMOTE_DIR} to ${COMMIT}"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "mkdir -p '$REMOTE_DIR' '$REMOTE_DIR/outputs/remote_sync_backups' && cd '$REMOTE_DIR' && git status --short > 'outputs/remote_sync_backups/status_${STAMP}.txt' 2>/dev/null || true && git diff > 'outputs/remote_sync_backups/diff_${STAMP}.patch' 2>/dev/null || true"
  rsync -az --delete \
    --exclude '.venv/' \
    --exclude 'data/' \
    --exclude 'outputs/' \
    --exclude 'checkpoints/' \
    --exclude 'wandb/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    "$REPO_ROOT"/ "$host:$REMOTE_DIR"/
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "cd '$REMOTE_DIR' && test \"\$(git rev-parse HEAD)\" = '$COMMIT' && test -x .venv/bin/python && .venv/bin/python --version && git status --short --branch"
}

case "$TARGET" in
  all)
    sync_one ripper
    sync_one foureyes
    sync_one ai
    ;;
  work|localhost)
    echo "Local worktree is already at ${COMMIT}: ${REPO_ROOT}"
    ;;
  ripper|foureyes|ai)
    sync_one "$TARGET"
    ;;
  *)
    echo "Usage: $0 [all|ripper|foureyes|ai|work] [remote_dir]" >&2
    exit 2
    ;;
esac
