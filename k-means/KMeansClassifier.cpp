//
// Created by emiliodallatorre on 06/12/25.
//

#include "KMeansClassifier.h"
#include "spdlog/spdlog.h"
#include <limits>
#include <iostream>

vector<int> KMeansClassifier::fit() {
    spdlog::info("Starting KMeans fit with {} clusters and max {} iterations", cluster_count, max_iterations);

    // Init centroids by selecting first k data points
    centroids_history.back().resize(cluster_count);
    for (int c = 0; c < cluster_count; c++) {
        centroids_history.back()[c] = data[c];
    }

    while (iteration_count < max_iterations) {
        // Step 1: Assign labels based on closest centroid
        vector<int> new_labels(data.size());

        for (size_t i = 0; i < data.size(); i++) {
            double min_dist = std::numeric_limits<double>::max();

            int best_cluster = -1;
            for (int c = 0; c < cluster_count; c++) {
                double dist = 0.0;

                for (size_t d = 0; d < data[i].size(); d++) {
                    double diff = data[i][d] - centroids_history.back()[c][d];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }

            new_labels[i] = best_cluster;
        }

        // Step 2: Update centroids
        vector<vector<double> > new_centroids(cluster_count, vector<double>(data[0].size(), 0.0));
        vector<int> counts(cluster_count, 0);
        for (size_t i = 0; i < data.size(); i++) {
            int cluster = new_labels[i];
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

        centroids_history.push_back(new_centroids);
        labels = new_labels;
        iteration_count++;
    }

    return labels;
}
