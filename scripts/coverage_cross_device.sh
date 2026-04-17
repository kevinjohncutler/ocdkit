#!/usr/bin/env bash
#
# Run ocdkit tests on Mac (MPS), Threadripper (CUDA), and
# Threadripper with CUDA disabled (CPU-only), then combine coverage.
# Notebook-based tests (IPython/Jupyter-dependent code) run on Mac only.
#
# Usage:  bash scripts/coverage_cross_device.sh
#
set -euo pipefail

MAC_ROOT="/Volumes/DataDrive/ocdkit"
REMOTE_ROOT="/home/kcutler/DataDrive/ocdkit"
REMOTE="kcutler@threadripper.local"
PYENV='export PATH="$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH"'
COV_DIR="$MAC_ROOT/.coverage_combined"

rm -rf "$COV_DIR"
mkdir -p "$COV_DIR"

echo "=== Mac (MPS) ==="
cd "$MAC_ROOT"
python -m coverage run --data-file="$COV_DIR/.coverage.mac" -m pytest tests/ -q

echo ""
echo "=== Threadripper (CUDA) ==="
ssh "$REMOTE" "$PYENV && cd $REMOTE_ROOT && python -m coverage run --data-file=/tmp/.coverage.cuda -m pytest tests/ -q --ignore=tests/test_plot_notebook.py"
scp "$REMOTE":/tmp/.coverage.cuda "$COV_DIR/.coverage.cuda"

echo ""
echo "=== Threadripper (CPU-only) ==="
ssh "$REMOTE" "$PYENV && cd $REMOTE_ROOT && CUDA_VISIBLE_DEVICES='' python -m coverage run --data-file=/tmp/.coverage.cpu -m pytest tests/ -q --ignore=tests/test_plot_notebook.py"
scp "$REMOTE":/tmp/.coverage.cpu "$COV_DIR/.coverage.cpu"

echo ""
echo "=== Combining coverage ==="
cd "$MAC_ROOT"
python -m coverage combine --data-file="$COV_DIR/.coverage" "$COV_DIR"/.coverage.*
python -m coverage report --data-file="$COV_DIR/.coverage" --show-missing
echo ""
python -m coverage html --data-file="$COV_DIR/.coverage" -d "$COV_DIR/htmlcov"
echo "HTML report: file://$COV_DIR/htmlcov/index.html"
