#ifndef K_MEANS_KMEANSCLASSIFIER_H
#define K_MEANS_KMEANSCLASSIFIER_H

#include <vector>


using namespace std;

class KMeansClassifier {
public:
    const int cluster_count;
    const int max_iterations;

    vector<vector<double> > data;
    vector<int> labels;

    // constructor that also accepts initial data
    KMeansClassifier(const vector<vector<double> > &init_data, const int clusters, const int max_iter = 300)
        : cluster_count(clusters), max_iterations(max_iter), data(init_data) {
    }

    vector<int> fit();

    int save_state(const char *filename);

    int load_state(const char *filename);

private:
    // Hereafter the internal state of the classifier
    vector<vector<vector<double> > > centroids_history {{}}; // history of centroids
    int iteration_count = 0;
};


#endif //K_MEANS_KMEANSCLASSIFIER_H
