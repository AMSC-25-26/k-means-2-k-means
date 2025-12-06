#ifndef K_MEANS_KMEANSCLASSIFIER_H
#define K_MEANS_KMEANSCLASSIFIER_H

#include <vector>

#define PERFORM_SEQUENTIAL_KMEANS true

using namespace std;

class KMeansClassifier {
public:
    const int cluster_count;
    const int max_iterations;

    vector<vector<double> > data;

    // constructor that also accepts initial data
    KMeansClassifier(const vector<vector<double> > &init_data, const int clusters, const int max_iter = 300)
        : cluster_count(clusters), max_iterations(max_iter), data(init_data) {
    }

    vector<int> fit();

    int save_state(const char *filename);

    int load_state(const char *filename);

private:
    // Hereafter the internal state of the classifier
    vector<vector<vector<double> > > centroids_history_sequential{{}}; // history of centroids
    vector<vector<vector<double> > > centroids_history_parallel{{}}; // history of centroids
    vector<int> labels_parallel, labels_sequential; // labels assigned to data points

    int iteration_count_sequential, iteration_count_parallel = 0;

    void set_initial_centroids();

    pair<vector<int>, int> fit_sequential();

    pair<vector<int>, int> fit_parallel();
};


#endif //K_MEANS_KMEANSCLASSIFIER_H
