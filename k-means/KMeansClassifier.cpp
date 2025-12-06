//
// Created by emiliodallatorre on 06/12/25.
//

#include "KMeansClassifier.h"
#include "spdlog/spdlog.h"
#include <limits>
#include <iostream>

void KMeansClassifier::set_initial_centroids() {
    centroids_history_sequential.back().resize(cluster_count);
    centroids_history_parallel.back().resize(cluster_count);

    for (int c = 0; c < cluster_count; c++) {
        centroids_history_sequential.back()[c] = data[c];
        centroids_history_parallel.back()[c] = data[c];
    }
}

vector<int> KMeansClassifier::fit() {
    // Init centroids by selecting first k data points
    set_initial_centroids();

    if constexpr (PERFORM_SEQUENTIAL_KMEANS) {
        pair<vector<int>, int> sequential_result = fit_sequential();
        labels_sequential = sequential_result.first;
        iteration_count_sequential = sequential_result.second;
    }
    pair<vector<int>, int> parallel_result = fit_parallel();
    labels_parallel = parallel_result.first;
    iteration_count_parallel = parallel_result.second;

    // Compare results
    if (labels_sequential != labels_parallel) {
        spdlog::warn("KMeans sequential and parallel results differ!");
    } else if constexpr (PERFORM_SEQUENTIAL_KMEANS) {
        spdlog::debug("KMeans sequential and parallel results match.");
    }

    return labels_parallel;
}

pair<vector<int>, int> KMeansClassifier::fit_parallel() {
    // Placeholder for parallel implementation
    spdlog::warn("Parallel KMeans not implemented yet, falling back to sequential.");

    return fit_sequential();
}

pair<vector<int>, int> KMeansClassifier::fit_sequential() {
    spdlog::info("Starting sequential KMeans fit with {} clusters and max {} iterations", cluster_count, max_iterations);

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

    spdlog::info("Finished sequential KMeans fit after {} iterations", iteration_count_sequential);
    return pair<vector<int>, int>(labels, iteration_count_sequential);
}
