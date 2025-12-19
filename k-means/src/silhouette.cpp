#include "silhouette.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

using namespace std;

namespace silhouette {
    // Calculate Euclidean distance between two points
    double euclidean_distance(const vector<double> &p1, const vector<double> &p2) {
        double sum = 0.0;
        // For each dimension, find the difference
        for (size_t i = 0; i < p1.size(); ++i) {
            const double diff = p1[i] - p2[i];
            sum += diff * diff; // Square the difference and add it
        }
        return sqrt(sum); // Take square root to get final distance
    }

    // Calculate silhouette coefficient for a single point
    // This measures how well one point fits in its cluster
    double calculate_point_silhouette(
        const size_t point_idx,
        const vector<vector<double> > &data,
        const vector<int> &labels,
        const int n_clusters
    ) {
        const auto &current_point = data[point_idx];
        const int point_cluster_id = labels[point_idx];

        // a: Average distance to other points in the same cluster
        double avg_intra_cluster_dist = 0.0;

        // b: Average distance to points in the nearest other cluster
        double min_avg_inter_cluster_dist = numeric_limits<double>::max();

        size_t num_points_in_same_cluster = 0;

        // Track distances to points in each cluster
        vector inter_cluster_distance_sums(n_clusters, 0.0);
        vector<size_t> inter_cluster_point_counts(n_clusters, 0);

        // Look at every other point in the dataset
        for (size_t i = 0; i < data.size(); ++i) {
            if (i == point_idx) continue; // Skip the point itself

            const double dist = euclidean_distance(current_point, data[i]);
            const int other_point_cluster_id = labels[i];

            // Check if this other point is in the same cluster
            if (other_point_cluster_id == point_cluster_id) {
                // Add distance to same-cluster total
                avg_intra_cluster_dist += dist;
                num_points_in_same_cluster++;
            } else {
                // Add distance to different-cluster totals
                inter_cluster_distance_sums[other_point_cluster_id] += dist;
                inter_cluster_point_counts[other_point_cluster_id]++;
            }
        }

        // Calculate average distance within the same cluster
        if (num_points_in_same_cluster > 0) {
            avg_intra_cluster_dist /= static_cast<double>(num_points_in_same_cluster);
        } else {
            // If the point is alone in its cluster, silhouette score is 0
            return 0.0;
        }

        // Find the nearest other cluster (smallest average distance)
        for (int cluster_id = 0; cluster_id < n_clusters; ++cluster_id) {
            if (cluster_id != point_cluster_id && inter_cluster_point_counts[cluster_id] > 0) {
                const double avg_dist = inter_cluster_distance_sums[cluster_id] / static_cast<double>(
                                            inter_cluster_point_counts[cluster_id]);
                if (avg_dist < min_avg_inter_cluster_dist) {
                    min_avg_inter_cluster_dist = avg_dist;
                }
            }
        }

        // If there are no other clusters, return 0
        if (min_avg_inter_cluster_dist == numeric_limits<double>::max()) {
            return 0.0;
        }

        // Silhouette formula: (b - a) / max(a, b)
        // If b > a: point is closer to its own cluster (good)
        // If a > b: point is closer to another cluster (bad)
        const double max_dist = max(avg_intra_cluster_dist, min_avg_inter_cluster_dist);
        if (max_dist == 0.0) {
            return 0.0;
        }

        return (min_avg_inter_cluster_dist - avg_intra_cluster_dist) / max_dist;
    }

    // Calculate complete silhouette score serially
    double complete_serial(
        const vector<vector<double> > &data,
        const vector<int> &labels,
        const int n_clusters
    ) {
        const size_t n_points = data.size();

        // Check if the input data is valid
        if (n_points == 0 || data.size() != labels.size()) {
            throw invalid_argument("Invalid input: empty data or size mismatch");
        }

        // Silhouette needs at least 2 clusters to compare
        if (n_clusters < 2) {
            return 0.0;
        }

        // Calculate silhouette for each point and add them up
        double total_silhouette = 0.0;
        for (size_t i = 0; i < n_points; ++i) {
            total_silhouette += calculate_point_silhouette(i, data, labels, n_clusters);
        }

        // Return the average silhouette score
        return total_silhouette / static_cast<double>(n_points);
    }

    // Calculate complete silhouette score in parallel
    double complete_parallel(
        const vector<vector<double> > &data,
        const vector<int> &labels,
        const int n_clusters
    ) {
        const size_t n_points = data.size();

        // Check if the input data is valid
        if (n_points == 0 || data.size() != labels.size()) {
            throw invalid_argument("Invalid input: empty data or size mismatch");
        }

        // Silhouette needs at least 2 clusters to compare
        if (n_clusters < 2) {
            return 0.0;
        }

        double total_silhouette = 0.0;

        // Use OpenMP to split the work automatically among threads
        // Each thread calculates some points and adds to total_silhouette
#pragma omp parallel for default(none) shared(data, labels, n_clusters, n_points) reduction(+:total_silhouette) schedule(dynamic)
        for (size_t i = 0; i < n_points; ++i) {
            total_silhouette += calculate_point_silhouette(i, data, labels, n_clusters);
        }

        // Return the average silhouette score
        return total_silhouette / static_cast<double>(n_points);
    }

    // Calculate approximate silhouette score serially using a random sample of the data
    double approximate_serial(
        const vector<vector<double> > &data,
        const vector<int> &labels,
        const int n_clusters,
        const size_t sample_size
    ) {
        const size_t n_points = data.size();

        // Check if the input data is valid
        if (n_points == 0 || data.size() != labels.size()) {
            throw invalid_argument("Invalid input: empty data or size mismatch");
        }

        // Silhouette needs at least 2 clusters to compare
        if (n_clusters < 2) {
            return 0.0;
        }

        // Don't sample more points than available
        const size_t actual_sample_size = min(sample_size, n_points);

        // Create a list of all point indices: [0, 1, 2, ..., n_points-1]
        vector<size_t> all_indices(n_points);
        iota(all_indices.begin(), all_indices.end(), 0);

        // Shuffle the list to get random order
        random_device rd;
        mt19937 gen(rd());
        ranges::shuffle(all_indices, gen);

        // Take the first 'actual_sample_size' points as random sample
        double total_silhouette = 0.0;
        for (size_t i = 0; i < actual_sample_size; ++i) {
            const size_t idx = all_indices[i];
            total_silhouette += calculate_point_silhouette(idx, data, labels, n_clusters);
        }

        // Return average based on sample size (not total points)
        return total_silhouette / static_cast<double>(actual_sample_size);
    }

    // Calculate approximate silhouette score in parallel using a random sample of the data
    double approximate_parallel(
        const vector<vector<double> > &data,
        const vector<int> &labels,
        const int n_clusters,
        const size_t sample_size
    ) {
        const size_t n_points = data.size();

        // Check if the input data is valid
        if (n_points == 0 || data.size() != labels.size()) {
            throw invalid_argument("Invalid input: empty data or size mismatch");
        }

        // Silhouette needs at least 2 clusters to compare
        if (n_clusters < 2) {
            return 0.0;
        }

        const size_t actual_sample_size = min(sample_size, n_points);

        // Create a list of all point indices: [0, 1, 2, ..., n_points-1]
        vector<size_t> all_indices(n_points);
        iota(all_indices.begin(), all_indices.end(), 0);

        // Shuffle the list to get random order
        random_device rd;
        mt19937 gen(rd());
        ranges::shuffle(all_indices, gen);

        // Take the first 'actual_sample_size' points as our random sample
        double total_silhouette = 0.0;

        // Use OpenMP to split the work automatically among threads
        // Each thread calculates some points and adds to total_silhouette
#pragma omp parallel for default(none) shared(data, labels, n_clusters, all_indices, actual_sample_size) reduction(+:total_silhouette) schedule(dynamic)
        for (size_t i = 0; i < actual_sample_size; ++i) {
            const size_t idx = all_indices[i];
            total_silhouette += calculate_point_silhouette(idx, data, labels, n_clusters);
        }

        // Return average based on sample size (not total points)
        return total_silhouette / static_cast<double>(actual_sample_size);
    }
}
