//
// Created by emiliodallatorre on 06/12/25.
//

#include "KMeansClassifier.h"
#include "spdlog/spdlog.h"
#include <limits>
#include <iostream>
#include <mpi.h>
#include "CentroidInitializer.h"
#include "omp.h"


using namespace std;

void KMeansClassifier::set_initial_centroids() {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        auto initial_centroids = CentroidInitializer::initialize_centroids(data, cluster_count, 42);

    // Salva i centroidi iniziali nella cronologia sequenziale e parallela
    centroids_history_sequential.back() = initial_centroids;
    centroids_history_parallel.back() = initial_centroids;

    // vettore lineare compatibile con MPI
    local_centroids_history.clear();
    for (int i = 0; i < cluster_count; i++) {
        for (int j = 0; j < data[0].size(); j++) {
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
    new_local_centroids_history = local_centroids_history;

    // Broadcast dei centroidi iniziali
    MPI_Bcast(local_centroids_history.data(), size_lch, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(new_local_centroids_history.data(), size_lch, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cluster_count,1,MPI_INT,0,MPI_COMM_WORLD);

    if (rank == 0) {
        spdlog::debug("Centroids initialized via K-Means++");
    }
}

vector<int> KMeansClassifier::fit() {
    // Init centroids by selecting first k data points
    set_initial_centroids();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double sequential_time = 0.0, parallel_time = 0.0;


    if (rank == 0) {
        if constexpr (PERFORM_SEQUENTIAL_KMEANS) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto sequential_result = fit_sequential();
        auto t1 = std::chrono::high_resolution_clock::now();
        sequential_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
        labels_sequential = sequential_result.first;
        iteration_count_sequential = sequential_result.second;
        spdlog::info("Sequential KMeans took {:.3f} ms", sequential_time);
    }
    }


    auto tp0 = std::chrono::high_resolution_clock::now();
    auto parallel_result = fit_parallel();
    auto tp1 = std::chrono::high_resolution_clock::now();
    parallel_time = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
    labels_parallel = parallel_result.first;
    iteration_count_parallel = parallel_result.second;

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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dimensions = data[0].size();

    // --- Prepara il vettore lineare dei dati solo rank 0 ---
    vector<double> data_single_vector;
    if (rank == 0) {
        data_single_vector.resize(data.size() * dimensions);
        for (size_t i = 0; i < data.size(); i++)
            for (size_t j = 0; j < dimensions; j++)
                data_single_vector[i * dimensions + j] = data[i][j];
    }

    // Condividi dimensioni dei dati
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Count totale degli elementi (double)
    int total_elements = rank == 0 ? static_cast<int>(data_single_vector.size()) : 0;
    MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int count = total_elements / size;
    int remainder = total_elements % size;

    // --- Preparazione indici e quantità per Scatterv ---
    vector<int> send_counts(size), send_start_idx(size);
    int start = 0;
    for (int i = 0; i < size; i++) {
        send_counts[i] = count + (i < remainder ? 1 : 0);
        send_start_idx[i] = start;
        start += send_counts[i];
    }

    // --- Alloca buffer locale ---
    int local_count = send_counts[rank];
    vector<double> local_data_vector(local_count);

    // Scatter dei dati
    MPI_Scatterv(rank == 0 ? data_single_vector.data() : nullptr,
                 send_counts.data(),
                 send_start_idx.data(),
                 MPI_DOUBLE,
                 local_data_vector.data(),
                 local_count,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    // --- Labels locali ---
    vector<int> local_labels(local_count / dimensions);

    int iteration_count_parallel = 0;

    vector<int>labels;

    spdlog::debug("Starting parallel KMeans in rank {}, working on {} datapoints, size: {}, count: {}, local count: {}, total elements: {}", rank,send_counts[rank],size,count,local_labels.capacity(),total_elements);


    // --- Ciclo KMeans ---
    while (iteration_count_parallel < max_iterations &&
           (iteration_count_parallel == 0 || local_centroids_history != new_local_centroids_history)) {

        // Assegna label ai dati locali
        for (size_t i = 0; i < local_labels.size(); i++) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = -1;
            for (int c = 0; c < cluster_count; c++) {
                double dist = 0.0;
                for (int d = 0; d < dimensions; d++)
                    dist += pow(local_data_vector[i * dimensions + d] - new_local_centroids_history[c * dimensions + d], 2);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            local_labels[i] = best_cluster;
        }

        // --- Gather delle label ---
        vector<int> recv_counts(size), recv_start_idx(size);
        start = 0;
        for (int i = 0; i < size; i++) {
            recv_counts[i] = send_counts[i] / dimensions;
            recv_start_idx[i] = start;
            start += recv_counts[i];
        }

        vector<int> new_labels(rank == 0 ? data.size() : 0);
        MPI_Gatherv(local_labels.data(),
                    local_labels.size(),
                    MPI_INT,
                    new_labels.data(),
                    recv_counts.data(),
                    recv_start_idx.data(),
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD);

        // --- Aggiorna centroidi (solo rank 0) ---
        if (rank == 0) {
            vector<vector<double>> new_centroids(cluster_count, vector<double>(dimensions, 0.0));
            vector<int> counts(cluster_count, 0);
            for (size_t i = 0; i < data.size(); i++) {
                int cluster = new_labels[i];
                for (int d = 0; d < dimensions; d++)
                    new_centroids[cluster][d] += data[i][d];
                counts[cluster]++;
            }
            for (int c = 0; c < cluster_count; c++)
                if (counts[c] > 0)
                    for (int d = 0; d < dimensions; d++)
                        new_centroids[c][d] /= counts[c];

            local_centroids_history = new_local_centroids_history;

            for (int c = 0; c < cluster_count; c++) {
                for (int d = 0; d < dimensions; d++) {
                    new_local_centroids_history[c * dimensions + d] = new_centroids[c][d];
                }
            }
        }

        // Broadcast dei centroidi aggiornati
        MPI_Bcast(local_centroids_history.data(), size_lch, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(new_local_centroids_history.data(), size_lch, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        labels = new_labels;
        iteration_count_parallel++;
        /*MPI_Barrier(MPI_COMM_WORLD);
        if (iteration_count_parallel == 1 || iteration_count_parallel == max_iterations-1) {
            spdlog::debug("local_centroids_history: {}", local_centroids_history[0]);
            spdlog::debug("new_local_centroids_history: {}", new_local_centroids_history[0]);
        }*/
    }
    /*spdlog::debug("local_centroids_history: {}", local_centroids_history[0]);
    spdlog::debug("new_local_centroids_history: {}", new_local_centroids_history[0]);
*/
    if (rank == 0) {
        spdlog::info("Finished parallel KMeans fit after {} iterations", iteration_count_parallel);
    }

    //MPI_Barrier(MPI_COMM_WORLD);
    return {labels, iteration_count_parallel};
}


pair<vector<int>, int> KMeansClassifier::fit_sequential() {
    spdlog::info("Starting sequential KMeans fit with {} clusters and max {} iterations", cluster_count,
                 max_iterations);

    vector<int> labels;
    int iteration_count_sequential = 0;
    // We stop when centroids do not change or we reach max iterations
    while (iteration_count_sequential < max_iterations && (
               iteration_count_sequential == 0 || centroids_history_sequential.back() != centroids_history_sequential[
                   centroids_history_sequential.size() - 2])) {
        // Step 1: Assign labels based on closest centroid
        vector<int> new_labels(data.size());

        for (size_t i = 0; i < data.size(); i++) {
            double min_dist = std::numeric_limits<double>::max();

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
        vector new_centroids(cluster_count, vector(data[0].size(), 0.0));
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
    return pair(labels, iteration_count_sequential);
}
