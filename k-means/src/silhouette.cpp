#include "silhouette.hpp"
#include <cmath>
#include <random>
#include <algorithm>

namespace silhouette
{
    // Calculate Euclidean distance between two points
    double euclidean_distance(const std::vector<double>& p1, const std::vector<double>& p2)
    {
        double sum = 0.0;
        for (size_t i = 0; i < p1.size(); ++i)
        {
            const double diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // Calculate silhouette coefficient for a single point
    double calculate_point_silhouette(
        const size_t point_idx,
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels,
        const int n_clusters
    )
    {
        const auto& current_point = data[point_idx];
        const int point_cluster_id = labels[point_idx];

        // a: Mean distance to other points in the same cluster (intra-cluster distance)
        double avg_intra_cluster_dist = 0.0;
        // b: Smallest mean distance to points in any other cluster (nearest-cluster distance)
        double min_avg_inter_cluster_dist = std::numeric_limits<double>::max();

        size_t num_points_in_same_cluster = 0;
        std::vector inter_cluster_distance_sums(n_clusters, 0.0);
        std::vector<size_t> inter_cluster_point_counts(n_clusters, 0);

        for (size_t i = 0; i < data.size(); ++i)
        {
            if (i == point_idx) continue;

            const double dist = euclidean_distance(current_point, data[i]);
            const int other_point_cluster_id = labels[i];

            if (other_point_cluster_id == point_cluster_id)
            {
                avg_intra_cluster_dist += dist;
                num_points_in_same_cluster++;
            }
            else
            {
                inter_cluster_distance_sums[other_point_cluster_id] += dist;
                inter_cluster_point_counts[other_point_cluster_id]++;
            }
        }

        if (num_points_in_same_cluster > 0)
        {
            avg_intra_cluster_dist /= static_cast<double>(num_points_in_same_cluster);
        }
        else
        {
            // If the point is the only one in its cluster, its silhouette score is 0.
            return 0.0;
        }

        for (int cluster_id = 0; cluster_id < n_clusters; ++cluster_id)
        {
            if (cluster_id != point_cluster_id && inter_cluster_point_counts[cluster_id] > 0)
            {
                const double avg_dist = inter_cluster_distance_sums[cluster_id] / static_cast<double>(
                    inter_cluster_point_counts[cluster_id]);
                if (avg_dist < min_avg_inter_cluster_dist)
                {
                    min_avg_inter_cluster_dist = avg_dist;
                }
            }
        }

        if (min_avg_inter_cluster_dist == std::numeric_limits<double>::max())
        {
            return 0.0;
        }

        // Silhouette score is (b - a) / max(a, b)
        const double max_dist = std::max(avg_intra_cluster_dist, min_avg_inter_cluster_dist);
        if (max_dist == 0.0)
        {
            return 0.0;
        }

        return (min_avg_inter_cluster_dist - avg_intra_cluster_dist) / max_dist;
    }

    double complete_serial(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels,
        const int n_clusters
    )
    {
        const size_t n_points = data.size();

        // Input validation
        if (n_points == 0 || data.size() != labels.size())
        {
            throw std::invalid_argument("Invalid input: empty data or size mismatch");
        }

        if (n_clusters < 2)
        {
            return 0.0; // Silhouette is undefined for single cluster
        }

        double total_silhouette = 0.0;
        for (size_t i = 0; i < n_points; ++i)
        {
            total_silhouette += calculate_point_silhouette(i, data, labels, n_clusters);
        }

        return total_silhouette / static_cast<double>(n_points);
    }

    double complete_parallel(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels,
        const int n_clusters
    )
    {
        const size_t n_points = data.size();

        // Input validation
        if (n_points == 0 || data.size() != labels.size())
        {
            throw std::invalid_argument("Invalid input: empty data or size mismatch");
        }

        if (n_clusters < 2)
        {
            return 0.0; // Silhouette is undefined for single cluster
        }

        double total_silhouette = 0.0;

        #pragma omp parallel for reduction(+:total_silhouette) schedule(dynamic)
        for (size_t i = 0; i < n_points; ++i)
        {
            total_silhouette += calculate_point_silhouette(i, data, labels, n_clusters);
        }

        return total_silhouette / static_cast<double>(n_points);
    }

    double approximate_serial(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels,
        const int n_clusters,
        const size_t sample_size
    )
    {
        const size_t n_points = data.size();

        // Input validation
        if (n_points == 0 || data.size() != labels.size())
        {
            throw std::invalid_argument("Invalid input: empty data or size mismatch");
        }

        if (n_clusters < 2)
        {
            return 0.0;
        }

        // Use smaller of sample_size or n_points
        const size_t actual_sample_size = std::min(sample_size, n_points);

        // Create a vector with all indices [0, 1, ..., n_points-1]
        std::vector<size_t> all_indices(n_points);
        std::iota(all_indices.begin(), all_indices.end(), 0);

        // Shuffle the indices to get a random order
        std::random_device rd;
        std::mt19937 gen(rd());
        std::ranges::shuffle(all_indices, gen);

        // The first `actual_sample_size` elements are our random sample.
        double total_silhouette = 0.0;
        for (size_t i = 0; i < actual_sample_size; ++i)
        {
            const size_t idx = all_indices[i];
            total_silhouette += calculate_point_silhouette(idx, data, labels, n_clusters);
        }

        return total_silhouette / static_cast<double>(actual_sample_size);
    }

    double approximate_parallel(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels,
        const int n_clusters,
        const size_t sample_size
    )
    {
        const size_t n_points = data.size();

        // Input validation
        if (n_points == 0 || data.size() != labels.size())
        {
            throw std::invalid_argument("Invalid input: empty data or size mismatch");
        }

        if (n_clusters < 2)
        {
            return 0.0;
        }

        // Use smaller of sample_size or n_points
        const size_t actual_sample_size = std::min(sample_size, n_points);

        // Create a vector with all indices [0, 1, ..., n_points-1]
        std::vector<size_t> all_indices(n_points);
        std::iota(all_indices.begin(), all_indices.end(), 0);

        // Shuffle the indices to get a random order
        std::random_device rd;
        std::mt19937 gen(rd());
        std::ranges::shuffle(all_indices, gen);

        // The first `actual_sample_size` elements are our random sample.
        double total_silhouette = 0.0;

        #pragma omp parallel for reduction(+:total_silhouette) schedule(dynamic)
        for (size_t i = 0; i < actual_sample_size; ++i)
        {
            const size_t idx = all_indices[i];
            total_silhouette += calculate_point_silhouette(idx, data, labels, n_clusters);
        }

        return total_silhouette / static_cast<double>(actual_sample_size);
    }
}
