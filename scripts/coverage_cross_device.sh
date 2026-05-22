#!/usr/bin/env bash
#
# Run ocdkit tests on a local machine (e.g. Mac/MPS), a remote machine
# (e.g. CUDA), and that same remote with CUDA disabled (CPU-only), then
# combine coverage. Notebook-based tests (IPython/Jupyter-dependent code)
# run on the local machine only.
#
# Configure via env vars (or a sourced ``scripts/coverage_cross_device.env``
# — see the example file at ``scripts/coverage_cross_device.env.example``):
#
#   OCDKIT_LOCAL_ROOT    absolute path to this repo on the local machine
#   OCDKIT_REMOTE_ROOT   absolute path to this repo on the remote machine
#   OCDKIT_REMOTE        ssh target for the remote machine (user@host)
#
# Usage:  bash scripts/coverage_cross_device.sh
#
set -euo pipefail

# Optionally source an untracked local config (gitignored).
_env_file="$(dirname "$0")/coverage_cross_device.env"
if [[ -f "$_env_file" ]]; then
    # shellcheck disable=SC1090
    source "$_env_file"
fi

LOCAL_ROOT="${OCDKIT_LOCAL_ROOT:?set OCDKIT_LOCAL_ROOT to the repo path on the local machine}"
REMOTE_ROOT="${OCDKIT_REMOTE_ROOT:?set OCDKIT_REMOTE_ROOT to the repo path on the remote machine}"
REMOTE="${OCDKIT_REMOTE:?set OCDKIT_REMOTE to the ssh target (user@host)}"
PYENV='export PATH="$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH"'
COV_DIR="$LOCAL_ROOT/.coverage_combined"

rm -rf "$COV_DIR"
mkdir -p "$COV_DIR"

echo "=== local (MPS / default device) ==="
cd "$LOCAL_ROOT"
python -m coverage run --data-file="$COV_DIR/.coverage.local" -m pytest tests/ -q

echo ""
echo "=== remote (CUDA) ==="
ssh "$REMOTE" "$PYENV && cd $REMOTE_ROOT && python -m coverage run --data-file=/tmp/.coverage.cuda -m pytest tests/ -q --ignore=tests/test_plot_notebook.py"
scp "$REMOTE":/tmp/.coverage.cuda "$COV_DIR/.coverage.cuda"

echo ""
echo "=== remote (CPU-only) ==="
ssh "$REMOTE" "$PYENV && cd $REMOTE_ROOT && CUDA_VISIBLE_DEVICES='' python -m coverage run --data-file=/tmp/.coverage.cpu -m pytest tests/ -q --ignore=tests/test_plot_notebook.py"
scp "$REMOTE":/tmp/.coverage.cpu "$COV_DIR/.coverage.cpu"

echo ""
echo "=== Combining coverage ==="
cd "$LOCAL_ROOT"
python -m coverage combine --data-file="$COV_DIR/.coverage" "$COV_DIR"/.coverage.*
python -m coverage report --data-file="$COV_DIR/.coverage" --show-missing
echo ""
python -m coverage html --data-file="$COV_DIR/.coverage" -d "$COV_DIR/htmlcov"
echo "HTML report: file://$COV_DIR/htmlcov/index.html"
