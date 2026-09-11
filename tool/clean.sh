#!/bin/sh
# Remove only known outputs from this checkout.
set -eu

cd "$(dirname "$0")/.."
rm -rf bin build bench/results .pytest_cache .ruff_cache
find mod test bench demo -type d \( -name __pycache__ -o -name '*.egg-info' \) \
    -prune -exec rm -rf {} +
