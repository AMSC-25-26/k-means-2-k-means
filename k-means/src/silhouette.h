#ifndef K_MEANS_2_K_MEANS_SILHOUETTE_H
#define K_MEANS_2_K_MEANS_SILHOUETTE_H

#include <vector>

namespace silhouette
{
    // Calculate Euclidean distance between two points
    double euclidean_distance(const std::vector<double>& p1, const std::vector<double>& p2);

    // Complete silhouette coefficient calculation (serial version)
    double complete_serial(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels,
        int n_clusters
    );

    // Complete silhouette coefficient calculation (parallel version)
    double complete_parallel(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels,
        int n_clusters
    );

    // Approximate silhouette coefficient calculation (serial version)
    double approximate_serial(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels,
        int n_clusters,
        size_t sample_size
    );

    // Approximate silhouette coefficient calculation (parallel version)
    double approximate_parallel(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels,
        int n_clusters,
        size_t sample_size
    );

    // Calculate silhouette coefficient for a single point
    double calculate_point_silhouette(
        size_t point_idx,
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels,
        int n_clusters
    );
}

#endif //K_MEANS_2_K_MEANS_SILHOUETTE_H
