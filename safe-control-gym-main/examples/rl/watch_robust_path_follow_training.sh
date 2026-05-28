#!/bin/bash

set -euo pipefail

OUT_DIR=${1:-./results/ppo_quadrotor_3D_robust_path_follow_v1}
LOG_FILE="${OUT_DIR}/std_out.txt"
CSV_FILE="${OUT_DIR}/logs/training_curve.csv"

echo "Watching training output:"
echo "  log: ${LOG_FILE}"
echo "  csv: ${CSV_FILE}"

if [ -f "${CSV_FILE}" ]; then
    echo
    echo "Latest CSV rows:"
    tail -5 "${CSV_FILE}"
fi

echo
echo "Live stdout log:"
touch "${LOG_FILE}"
tail -f "${LOG_FILE}"
