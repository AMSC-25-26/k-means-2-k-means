#ifndef CENTROIDINITIALAZER_H
#define CENTROIDINITIALAZER_H

#include <vector>
#include <random>

enum class InitializationMethod
{
    random,
    kmeansPlusPlus
};

class CentroidInitializer
{
public:
    static std::vector<std::vector<double>> initialize_centroids(
        const std::vector<std::vector<double>>& data,
        int k,
        // InitializationMethod method = InitializationMethod::kmeansPlusPlus,
        int seed = 42
    );

private:
    static std::vector<std::vector<double>> init_random(
        const std::vector<std::vector<double>>& data,
        int k,
        std::mt19937& gen
    );

    static std::vector<std::vector<double>> init_kmeans_plus_plus(
        const std::vector<std::vector<double>>& data,
        int k,
        std::mt19937& gen
    );
};

#endif // CENTROIDINITIALAZER_H
