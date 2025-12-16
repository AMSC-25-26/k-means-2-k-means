#!/usr/bin/env bash
set -euo pipefail

DATASET=./data/dataset-iris.csv
K=3
GT=./data/wine-clusters.csv
BUILD=./cmake-build-debug/k_means

for n in 1 2 3 4 5 6 7 8; do
  echo "----- Running with n=${n} -----"
  start=$(date +%s.%N)

  # Run the program but suppress all output (stdout and stderr) so we only
  # measure elapsed time and the exit code below.
  mpiexec -np "${n}" "${BUILD}" "${DATASET}" /dev/null "${K}" "${GT}" > /dev/null #2>&1
  rc=$?

  end=$(date +%s.%N)
  elapsed=$(awk "BEGIN {printf \"%.6f\", ${end} - ${start}}")

  echo "n=${n}, elapsed_seconds=${elapsed}, exit_code=${rc}"
done