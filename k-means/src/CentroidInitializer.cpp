// CentroidInitializer.cpp
#include "CentroidInitializer.hpp"
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <set>
#include "spdlog/spdlog.h"

using namespace std;

vector<vector<double>> CentroidInitializer::initialize_centroids(
    const vector<vector<double>>& data,
    int k,
    // InitializationMethod method,
    int seed
)
{
    mt19937 gen(seed); //Generatore di numeri casuali tramite Mersene Twister

    if (data.empty() || k <= 0 || k > data.size())
    {
        spdlog::error("Invalid parameters for centroid initialization");
        return {};
    }
    /* NECESSARIO PER IL METODO DI INIZIALIZZAZIONE DEI CENTROIDI CASUALI, NOI USEREMO SOLO K-MEANS++
        if (method == InitializationMethod::Random) {
            spdlog::info("Initializing centroids using Random strategy");
            return init_random(data, k, gen);
        } else {*/

    spdlog::info("Initializing centroids using K-Means++ strategy");
    return init_kmeans_plus_plus(data, k, gen);
    /*
    }
    */
}

/*
// Scegliere k indici a caso tramite un generatore di numeri casuali
vector<vector<double>> CentroidInitializer::init_random(
    const vector<vector<double>>& data, 
    int k, 
    mt19937& gen
) {
    vector<vector<double>> centroids;
    vector<int> indices(data.size());
    // Riempi indices con 0, 1, 2, ...
    for(size_t i = 0; i < data.size(); ++i) indices[i] = i;

    // Mischia
    shuffle(indices.begin(), indices.end(), gen);

    // Prendi i primi k
    for(int i = 0; i < k; ++i) {
        centroids.push_back(data[indices[i]]);
    }
    return centroids;
}
*/


// Miglior opzione per inizializzare: K-Means++
// 1. Scegliamo il primo centroide a caso.
// 2. Per ogni punto, calcoliamo la distanza (quadrata) dal centroide più vicino già scelto.
// 3. Scegliamo il prossimo centroide con probabilità proporzionale a quella distanza.
vector<vector<double>> CentroidInitializer::init_kmeans_plus_plus(
    const vector<vector<double>>& data,
    int k,
    mt19937& gen
)
{
    vector<vector<double>> centroids;
    size_t n_samples = data.size();

    // Scegliamo il primo centroide a caso
    uniform_int_distribution<> dis(0, n_samples - 1);
    centroids.push_back(data[dis(gen)]);

    // Array per memorizzare la distanza minima quadrata di ogni punto dal centroide più vicino
    vector<double> min_dist_sq(n_samples, numeric_limits<double>::max());

    // Troviamo i restanti k-1 centroidi tramite un loop
    for (int i = 1; i < k; ++i)
    {
        double sum_dist_sq = 0.0;

        // Aggiungiamo le distanze minime
        for (size_t j = 0; j < n_samples; ++j)
        {
            // Calcoliamo la distanza dall'ultimo centroide aggiunto, e aggiorniamo la distanza minima se necessario
            double dist_sq = 0.0;
            for (size_t dim = 0; dim < data[j].size(); ++dim)
            {
                double diff = data[j][dim] - centroids.back()[dim];
                dist_sq += diff * diff;
            }

            // Manteniamo la minima distanza trovata finora tra tutti i centroidi
            if (dist_sq < min_dist_sq[j])
            {
                min_dist_sq[j] = dist_sq;
            }
            sum_dist_sq += min_dist_sq[j];
        }

        // Selezioniamo il prossimo centroide con probabilità pesata, ovvero più è lontano, più è probabile che venga scelto
        uniform_real_distribution<> prob_dis(0.0, sum_dist_sq);
        double random_val = prob_dis(gen);
        double cumulative_sum = 0.0;
        int selected_index = -1;

        for (size_t j = 0; j < n_samples; ++j)
        {
            cumulative_sum += min_dist_sq[j];
            if (cumulative_sum >= random_val)
            {
                selected_index = j;
                break;
            }
        }

        // Fallback nel caso di errori di arrotondamento (raro)
        if (selected_index == -1) selected_index = n_samples - 1;

        centroids.push_back(data[selected_index]);
    }

    return centroids;
}
