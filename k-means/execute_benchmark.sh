#!/usr/bin/env bash
set -euo pipefail

DATASET=./data/dataset-iris.csv
K=3
GT=./data/wine-clusters.csv
BUILD=./cmake-build-debug/k_means

mkdir -p logs

for n in 1 2 3 4 5 6 7 8; do
  echo "----- Running with n=${n} -----"
  start=$(date +%s.%N)

  # Capture mpiexec stdout+stderr into a temporary file. Use `|| rc=$?` so we
  # still capture the program's exit code even though `set -e` is enabled.
  tmpfile=$(mktemp)
  rc=0
  mpiexec -np "${n}" "${BUILD}" "${DATASET}" /dev/null "${K}" "${GT}" > "${tmpfile}" 2>&1 || rc=$?

  end=$(date +%s.%N)
  elapsed=$(awk "BEGIN {printf \"%.6f\", ${end} - ${start}}")

  # If the program failed (rc != 0), persist the output and print it.
  if [ "$rc" -ne 0 ]; then
    saved="logs/run-${n}.log"
    mv "${tmpfile}" "${saved}"
    echo "n=${n}, elapsed_seconds=${elapsed}, exit_code=${rc}"
    echo "==== mpiexec output (saved to ${saved}) ===="
    cat "${saved}"
  else
    # Success: discard the temporary output
    rm -f "${tmpfile}"
    echo "n=${n}, elapsed_seconds=${elapsed}, exit_code=${rc}"
  fi

done