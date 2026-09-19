#!/usr/bin/env bash
# Full verification pass: tests, detection report, mutation probe.
#
#   ./scripts/run_verification_suite.sh
#
# Exits non-zero if any stage fails. Safe to run with no network.

set -euo pipefail

cd "$(dirname "$0")/.."

# Deterministic arithmetic requires single-threaded BLAS reductions.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

echo "=== 1/3  test suite ==============================================="
python3 -m pytest tests -q

echo
echo "=== 2/3  detection report ========================================="
python3 scripts/verification_report.py

echo
echo "=== 3/3  mutation probe ==========================================="
python3 scripts/mutation_probe.py

echo
echo "All stages passed."
