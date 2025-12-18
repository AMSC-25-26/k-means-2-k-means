k-means (MPI)
=================================

Purpose
-------
Small MPI-enabled K‑Means implementation with a clustering-quality report (silhouette score). This repository contains
the K‑Means algorithm, silhouette implementations (exact, parallel, approximate), and a small benchmarking harness. This
README explains how to build, run, and benchmark the code and gives a concise technical note on the silhouette score and
its implementation.

Quick build
-----------
(Recommended out-of-source build)

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
# the built binary (debug/rel) will be available as <build-dir>/k_means
```

If you already have the provided debug build, the binary is at `cmake-build-debug/k_means`.

Run (usage)
-----------
The program expects 3 command-line arguments (MPI handles parallelism):

```
mpiexec -np <N_PROCS> <path/to/k_means> <input_csv> <output_csv> <cluster_count>
```

Example (iris, 3 clusters, single process):

```bash
mpiexec -np 1 cmake-build-debug/k_means data/dataset-iris.csv output.csv 3
```

Behavior:

- `output.csv` will be written by the program and contains predicted cluster labels.
- Silhouette score is computed and logged only on rank 0.
- Exit code 0 indicates success; non‑zero indicates failure.

Benchmark
---------
A simple benchmark harness is provided: `execute_benchmark.sh`.

- It runs multiple process-counts (configurable within the script).
- For each run it measures elapsed wall‑time and captures the program output to a temporary file.
- If a run succeeds (exit code 0) the temporary output is discarded and the script prints only
  `n=..., elapsed_seconds=..., exit_code=0`.
- If a run fails (exit code != 0) the script moves the captured output to `logs/run-<n>.log` and prints the log content
  for debugging.

To run the benchmark:

```bash
chmod +x execute_benchmark.sh
./execute_benchmark.sh
```

Results (example: iris dataset)
-------------------------------
We ran a small benchmark on `data/dataset-iris.csv` with 3 clusters. The following table summarizes elapsed wall time,
exit code and silhouette score (averaged) for each run. Full captured outputs are saved under `results/`.

| n (MPI procs) | elapsed_seconds | exit_code | silhouette | log file                |
|--------------:|----------------:|----------:|-----------:|-------------------------|
|             1 |        1.974626 |         0 |   0.552592 | results/out-bench-1.log |
|             2 |        0.239624 |         0 |   0.552592 | results/out-bench-2.log |
|             4 |        0.250402 |         0 |   0.262306 | results/out-bench-4.log |

See `results/summary.csv` for a machine-readable summary and `results/out-bench-<n>.log` for the full program output.

Silhouette score — short technical note
-------------------------------------
What it measures

- For a point i, the silhouette s(i) ∈ [-1, +1] shows how well i fits its assigned cluster.
    - s(i) ≈ +1: well-assigned (close to its cluster, far from others)
    - s(i) ≈ 0: on the border between clusters
    - s(i) ≈ -1: likely misassigned (closer to another cluster)

Per-point computation

- a(i) = average distance from i to all other points in its cluster (intra-cluster).
- For every other cluster C, compute average distance from i to points in C; b(i) = minimum such average (nearest other
  cluster).
- s(i) = (b(i) - a(i)) / max(a(i), b(i)). If max(a,b) == 0, s(i) is defined as 0.

Overall score

- The reported silhouette score is the average s(i) over all points.

Files & implementation notes

- `src/silhouette.hpp` / `src/silhouette.cpp` — full implementation.
    - `euclidean_distance(...)` — Euclidean distance used as metric.
    - `calculate_point_silhouette(...)` — computes s(i) exactly using the formula above.
    - `complete_serial(...)` — exact silhouette computed for every point (single-threaded).
    - `complete_parallel(...)` — exact silhouette computed with OpenMP (parallelized per-point).
    - `approximate_serial(...)` / `approximate_parallel(...)` — randomly sample points (faster, approximate).
- `src/main.cpp` — runs K‑Means, saves labels, and calls `silhouette::complete_parallel` on rank 0; logs the value and
  emits a warning when the score is below 0.25 (heuristic). Change that threshold inside `main.cpp` if desired.

Complexity & tradeoffs

- `complete_serial`: O(n^2 * d) time and O(1) extra memory (n = #points, d = dimension). Accurate but quadratic;
  expensive for large n.
- `complete_parallel`: same asymptotic cost, but uses OpenMP to utilize multiple CPU cores and reduce wall-clock time.
- `approximate_*`: O(m * n * d) where m is sample size (m << n). Reduces runtime at cost of accuracy.

Practical advice

- For small datasets (iris, wine) `complete_serial` is fine and exact.
- For large datasets use `approximate_*` or `complete_parallel` compiled with OpenMP support (`-fopenmp`).
- If you change the silhouette function or parallelization, re-run CMake to pick up flags (if using FindOpenMP in
  CMake).

Logging & debugging

- The program uses `spdlog` for messages. Silhouette score and warnings are printed on rank 0.
- Benchmark logs for failing runs are saved into `logs/run-<n>.log` by `execute_benchmark.sh`.

Reproducibility & test datasets

- Example datasets in `data/`: `dataset-iris.csv` and `dataset-wine.csv`.
- Provided expected cluster CSVs: `iris-clusters.csv`, `wine-clusters.csv`.

Contact / notes for the professor

- This is the midterm project implementation: MPI K‑Means and silhouette-based evaluation.

---
Created for the course midterm.
