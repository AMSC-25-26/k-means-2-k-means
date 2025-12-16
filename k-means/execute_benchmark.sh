#!/usr/bin/env bash
set -euo pipefail

DATASET=./data/dataset-wine.csv
K=11
GT=./data/wine-clusters.csv
BUILD=./cmake-build-debug/k_means

for n in 1 2 4 8; do
  echo "----- Running with n=${n} -----"
  start=$(date +%s.%N)

  # pass /dev/stdout so the program prints its output to the terminal
  mpiexec -n "${n}" "${BUILD}" "${DATASET}" /dev/null "${K}" "${GT}"
  rc=$?

  end=$(date +%s.%N)
  elapsed=$(awk "BEGIN {printf \"%.6f\", ${end} - ${start}}")

  echo "n=${n}, elapsed_seconds=${elapsed}, exit_code=${rc}"
done