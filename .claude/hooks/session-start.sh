#!/bin/bash
set -euo pipefail

# Only run in remote (cloud) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Install Python 3.13 and sync dependencies using uv
# uv sync handles Python installation (if needed) and dependency resolution
# from pyproject.toml + uv.lock in a single idempotent step
uv sync

# Make ruff available on PATH by adding the venv bin directory
echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
