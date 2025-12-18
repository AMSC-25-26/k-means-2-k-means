//
// Created by emiliodallatorre on 06/12/25.
//

#include "KMeansClassifier.hpp"
#include "spdlog/spdlog.h"
#include <limits>
#include <iostream>
#include <mpi.h>
#include "CentroidInitializer.hpp"


using namespace std;

void KMeansClassifier::set_initial_centroids() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Broadcast cluster_count first
    MPI_Bcast(&cluster_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (data.empty() || data[0].empty()) {
            spdlog::error("Data is empty, cannot initialize centroids");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const int dimensions = static_cast<int>(data[0].size());
        const auto initial_centroids = CentroidInitializer::initialize_centroids(data, cluster_count, 42);

        // Salva i centroidi iniziali nella cronologia sequenziale e parallela
        centroids_history_sequential.back() = initial_centroids;
        centroids_history_parallel.back() = initial_centroids;

        // vettore lineare compatibile con MPI
        local_centroids_history.clear();
        for (int i = 0; i < cluster_count; i++) {
            for (int j = 0; j < dimensions; j++) {
                local_centroids_history.push_back(initial_centroids[i][j]);
            }
        }

        // Imposta dimensione  dei centroidi
        size_lch = static_cast<int>(local_centroids_history.size());
    }

    // Broadcast della dimensione
    MPI_Bcast(&size_lch, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Ridimensiona vettori
    if (local_centroids_history.size() != size_lch)
        local_centroids_history.resize(size_lch);
    if (new_local_centroids_history.size() != size_lch)
        new_local_centroids_history.resize(size_lch);

    // Broadcast dei centroidi iniziali
    MPI_Bcast(local_centroids_history.data(), size_lch, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    new_local_centroids_history = local_centroids_history;

    if (rank == 0) {
        spdlog::debug("Centroids initialized via K-Means++");
    }
}

vector<int> KMeansClassifier::fit() {
    // Init centroids by selecting first k data points
    set_initial_centroids();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double parallel_time = 0.0;

    if (rank == 0) {
        if constexpr (PERFORM_SEQUENTIAL_KMEANS) {
            double sequential_time = 0.0;
            const auto t0 = chrono::high_resolution_clock::now();
            auto [fst, snd] = fit_sequential();
            const auto t1 = chrono::high_resolution_clock::now();
            sequential_time = chrono::duration<double, milli>(t1 - t0).count();
            labels_sequential = fst;
            iteration_count_sequential = snd;
            spdlog::info("Sequential KMeans took {:.3f} ms", sequential_time);
        }
    }

    const auto tp0 = chrono::high_resolution_clock::now();
    auto [fst, snd] = fit_parallel();
    const auto tp1 = chrono::high_resolution_clock::now();
    parallel_time = chrono::duration<double, milli>(tp1 - tp0).count();
    labels_parallel = fst;
    iteration_count_parallel = snd;

    if (rank == 0) {
        spdlog::info("Parallel KMeans took {:.3f} ms", parallel_time);
    }


    if (rank == 0) {
        if (labels_sequential != labels_parallel) {
            spdlog::warn("KMeans sequential and parallel results differ!");
        } else if constexpr (PERFORM_SEQUENTIAL_KMEANS) {
            spdlog::debug("KMeans sequential and parallel results match.");
        }
    }
    // Compare results
    return labels_parallel;
}

pair<vector<int>, int> KMeansClassifier::fit_parallel() {
    // Get MPI rank (process ID) and size (total number of processes)
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the number of dimensions from the data
    int dimensions = 0;
    if (rank == 0) {
        dimensions = static_cast<int>(data[0].size());
    }

    // Prepare a flat vector of all data points (only on rank 0)
    // This converts 2D data into 1D: [[x1,y1], [x2,y2]] -> [x1,y1,x2,y2]
    vector<double> data_single_vector;
    int num_points = 0;
    if (rank == 0) {
        num_points = static_cast<int>(data.size());
        data_single_vector.resize(num_points * dimensions);
        for (size_t i = 0; i < data.size(); i++)
            for (int j = 0; j < dimensions; j++)
                data_single_vector[i * dimensions + j] = data[i][j];
    }

    // Share the dimensions and number of points to all processes
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate how many complete data points each process will get
    // Complete data points must be distributed, not individual values
    // This ensures a point [x,y,z] stays together on one process instead of being split
    const int points_per_proc = num_points / size;
    const int remainder_points = num_points % size;

    // Prepare arrays for MPI_Scatterv
    // send_counts[i] = how many doubles process i will receive
    // send_start_idx[i] = starting index in the data array for process i
    vector<int> send_counts(size), send_start_idx(size);
    int start = 0;
    for (int i = 0; i < size; i++) {
        // Some processes get one extra point if there's a remainder
        const int points_for_this_proc = points_per_proc + (i < remainder_points ? 1 : 0);

        // Convert number of points to number of doubles (multiply by dimensions)
        send_counts[i] = points_for_this_proc * dimensions;
        send_start_idx[i] = start;
        start += send_counts[i];
    }

    // Allocate local buffer to receive this process's share of data
    const int local_count = send_counts[rank];
    vector<double> local_data_vector(local_count);

    // Scatter: distribute data from rank 0 to all processes
    // Each process gets its chunk of complete data points
    MPI_Scatterv(rank == 0 ? data_single_vector.data() : nullptr,
                 send_counts.data(),
                 send_start_idx.data(),
                 MPI_DOUBLE,
                 local_data_vector.data(),
                 local_count,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    // Calculate how many complete data points this process has
    const int local_num_points = local_count / dimensions;
    vector<int> local_labels(local_num_points);

    iteration_count_parallel = 0;

    vector<int> labels;

    spdlog::debug(
        "Starting parallel KMeans in rank {}, working on {} datapoints, dimensions: {}, local count: {}, total points: {}",
        rank, local_num_points, dimensions, local_count, num_points);

    bool converged = false;

    // Main K-Means loop
    while (iteration_count_parallel < max_iterations && !converged) {
        // Step 1: Assign cluster labels to local data points
        // Each process works on its own chunk of data
        for (int i = 0; i < local_num_points; i++) {
            double min_dist = numeric_limits<double>::max();
            int best_cluster = -1;
            // Find the closest centroid for this data point
            for (int c = 0; c < cluster_count; c++) {
                double dist = 0.0;
                for (int d = 0; d < dimensions; d++)
                    dist += pow(local_data_vector[i * dimensions + d] - new_local_centroids_history[c * dimensions + d],
                                2);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            local_labels[i] = best_cluster;
        }

        // Step 2: Gather all labels back to rank 0
        // Prepare receive counts (convert from doubles back to number of points)
        vector<int> recv_counts(size), recv_start_idx(size);
        start = 0;
        for (int i = 0; i < size; i++) {
            recv_counts[i] = send_counts[i] / dimensions;
            recv_start_idx[i] = start;
            start += recv_counts[i];
        }

        // Collect all labels on rank 0
        vector<int> new_labels(rank == 0 ? num_points : 0);
        MPI_Gatherv(local_labels.data(),
                    local_num_points,
                    MPI_INT,
                    new_labels.data(),
                    recv_counts.data(),
                    recv_start_idx.data(),
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD);

        // Step 3: Update centroids (only rank 0 does this)
        if (rank == 0) {
            // Save current centroids to check for convergence later
            local_centroids_history = new_local_centroids_history;

            // Calculate new centroids based on the mean of assigned points
            vector<vector<double> > new_centroids(cluster_count, vector<double>(dimensions, 0.0));
            vector<int> counts(cluster_count, 0);
            for (int i = 0; i < num_points; i++) {
                const int cluster = new_labels[i];
                for (int d = 0; d < dimensions; d++)
                    new_centroids[cluster][d] += data[i][d];
                counts[cluster]++;
            }
            // Divide by count to get the mean
            for (int c = 0; c < cluster_count; c++)
                if (counts[c] > 0)
                    for (int d = 0; d < dimensions; d++)
                        new_centroids[c][d] /= counts[c];

            // Flatten the new centroids into a 1D vector for broadcasting
            for (int c = 0; c < cluster_count; c++) {
                for (int d = 0; d < dimensions; d++) {
                    new_local_centroids_history[c * dimensions + d] = new_centroids[c][d];
                }
            }
        }

        // Step 4: Broadcast new centroids to all processes
        // This must happen before checking convergence
        // All processes need the same centroids for the next iteration
        MPI_Bcast(new_local_centroids_history.data(), size_lch, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Step 5: Check if centroids have converged (only on rank 0)
        // Check happens after broadcast to ensure consistency
        if (rank == 0) {
            // Compare old centroids with new centroids
            // If they're the same, the algorithm has converged
            converged = (iteration_count_parallel > 0) &&
                        (local_centroids_history == new_local_centroids_history);
        }

        // Step 6: Share convergence status with all processes
        MPI_Bcast(&converged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        // Only rank 0 keeps the full labels array
        if (rank == 0) {
            labels = new_labels;
        }
        iteration_count_parallel++;
    }

    if (rank == 0) {
        spdlog::info("Finished parallel KMeans fit after {} iterations", iteration_count_parallel);
    }

    return {labels, iteration_count_parallel};
}


pair<vector<int>, int> KMeansClassifier::fit_sequential() {
    spdlog::info("Starting sequential KMeans fit with {} clusters and max {} iterations", cluster_count,
                 max_iterations);

    vector<int> labels;
    iteration_count_sequential = 0;
    // We stop when centroids do not change, or we reach max iterations
    while (iteration_count_sequential < max_iterations && (
               iteration_count_sequential == 0 || centroids_history_sequential.back() != centroids_history_sequential[
                   centroids_history_sequential.size() - 2])) {
        // Step 1: Assign labels based on closest centroid
        vector<int> new_labels(static_cast<int>(data.size()));

        for (size_t i = 0; i < data.size(); i++) {
            double min_dist = numeric_limits<double>::max();

            int best_cluster = -1;
            for (int c = 0; c < cluster_count; c++) {
                double dist = 0.0;

                for (size_t d = 0; d < data[i].size(); d++) {
                    const double diff = data[i][d] - centroids_history_sequential.back()[c][d];
                    dist += diff * diff;
                } // This is the squared Euclidean distance
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }

            new_labels[i] = best_cluster;
        }

        // Step 2: Update centroids
        vector new_centroids(cluster_count, vector(static_cast<int>(data[0].size()), 0.0));
        vector counts(cluster_count, 0);
        for (size_t i = 0; i < data.size(); i++) {
            const int cluster = new_labels[i];
            for (size_t d = 0; d < data[i].size(); d++) {
                new_centroids[cluster][d] += data[i][d];
            }
            counts[cluster]++;
        }
        for (int c = 0; c < cluster_count; c++) {
            if (counts[c] > 0) {
                for (size_t d = 0; d < new_centroids[c].size(); d++) {
                    new_centroids[c][d] /= counts[c];
                }
            }
        }

        centroids_history_sequential.push_back(new_centroids);
        labels = new_labels;
        iteration_count_sequential++;
    }
    //spdlog::debug("centroid: {}", centroids_history_sequential[centroids_history_sequential.size()-1][0][0]);

    spdlog::info("Finished sequential KMeans fit after {} iterations", iteration_count_sequential);
    return {labels, iteration_count_sequential};
}
