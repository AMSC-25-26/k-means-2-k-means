#ifndef K_MEANS_2_K_MEANS_SILHOUETTE_H
#define K_MEANS_2_K_MEANS_SILHOUETTE_H

#include <vector>

/**
 * Silhouette Coefficient Functions
 *
 * The silhouette coefficient measures how well data points fit in their clusters.
 * - Values close to +1 mean the point is well-matched to its own cluster
 * - Values close to 0 mean the point is on the border between two clusters
 * - Values close to -1 mean the point might be in the wrong cluster
 */
namespace silhouette {
    /**
     * Calculate the straight-line distance between two points
     *
     * @param p1 First point
     * @param p2 Second point
     * @return The distance between the two points
     */
    double euclidean_distance(const std::vector<double> &p1, const std::vector<double> &p2);

    /**
     * Calculate silhouette score for all data points (single-threaded).
     * This version uses one CPU core and checks every point (high accuracy, slower).
     *
     * @param data All data points
     * @param labels Which cluster each point belongs to
     * @param n_clusters Total number of clusters
     * @return Average silhouette score (between -1 and +1)
     */
    double complete_serial(
        const std::vector<std::vector<double> > &data,
        const std::vector<int> &labels,
        int n_clusters
    );

    /**
     * Calculate silhouette score for all data points (multithreaded).
     * This version parallelizes using OpenMP for faster calculation (high accuracy, slower).
     *
     * @param data All data points
     * @param labels Which cluster each point belongs to
     * @param n_clusters Total number of clusters
     * @return Average silhouette score (between -1 and +1)
     */
    double complete_parallel(
        const std::vector<std::vector<double> > &data,
        const std::vector<int> &labels,
        int n_clusters
    );

    /**
     * Calculate approximate silhouette score using random sample (single-threaded).
     * This version is faster because it only checks some points, not all.
     *
     * @param data All data points
     * @param labels Which cluster each point belongs to
     * @param n_clusters Total number of clusters
     * @param sample_size How many random points to check
     * @return Estimated average silhouette score (between -1 and +1)
     */
    double approximate_serial(
        const std::vector<std::vector<double> > &data,
        const std::vector<int> &labels,
        int n_clusters,
        size_t sample_size
    );

    /**
     * Calculate approximate silhouette score using random sample (multithreaded).
     * This version is the fastest: uses multiple cores and checks only some points.
     *
     * @param data All data points
     * @param labels Which cluster each point belongs to
     * @param n_clusters Total number of clusters
     * @param sample_size How many random points to check
     * @return Estimated average silhouette score (between -1 and +1)
     */
    double approximate_parallel(
        const std::vector<std::vector<double> > &data,
        const std::vector<int> &labels,
        int n_clusters,
        size_t sample_size
    );

    /**
     * Calculate silhouette score for one single point
     *
     * @param point_idx Index of the point to check
     * @param data All data points
     * @param labels Which cluster each point belongs to
     * @param n_clusters Total number of clusters
     * @return Silhouette score for this point (between -1 and +1)
     */
    double calculate_point_silhouette(
        size_t point_idx,
        const std::vector<std::vector<double> > &data,
        const std::vector<int> &labels,
        int n_clusters
    );
}

#endif //K_MEANS_2_K_MEANS_SILHOUETTE_H
