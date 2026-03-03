#!/bin/bash
set -euo pipefail

# Only run in remote (cloud) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Install LaTeX packages required by matplotlib for rendering TeX labels in figures
apt-get update -qq && apt-get install -y -qq texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvipng cm-super > /dev/null 2>&1

# Install Python 3.13 and sync dependencies using uv
# uv sync handles Python installation (if needed) and dependency resolution
# from pyproject.toml + uv.lock in a single idempotent step
uv sync

# Make ruff available on PATH by adding the venv bin directory
echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
