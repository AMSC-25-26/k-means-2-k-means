#ifndef CENTROIDINITIALAZER_H
#define CENTROIDINITIALAZER_H

#include <vector>
#include <random>

using namespace std;

enum class InitializationMethod {
    random,
    kmeansPlusPlus
};

class CentroidInitializer {
public:
    static vector<vector<double> > initialize_centroids(
        const vector<vector<double> > &data,
        int k,
        // InitializationMethod method = InitializationMethod::kmeansPlusPlus,
        int seed = 42
    );

private:
    static vector<vector<double> > init_random(
        const vector<vector<double> > &data,
        int k,
        mt19937 &gen
    );

    static vector<vector<double> > init_kmeans_plus_plus(
        const vector<vector<double> > &data,
        int k,
        mt19937 &gen
    );
};

#endif // CENTROIDINITIALAZER_H
