#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
exec python "$REPO/scripts/run_tiny_comparison.py" "$@"
