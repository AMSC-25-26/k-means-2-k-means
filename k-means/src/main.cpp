#include <vector>
#include <string>
#include <spdlog/spdlog.h>
#include "data_io.cpp"
#include "KMeansClassifier.hpp"
#include "silhouette.hpp"
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[]) {
    spdlog::set_level(spdlog::level::debug);

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Debug print MPI info
    if (rank == 0) {
        spdlog::debug("MPI initialized with {} available threads", size);
    }

    if (argc < 4) {
        spdlog::error("Usage: {} <input_csv> <output_csv> <cluster_count>", argv[0]);
        return 1;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];

    int cluster_count = 0;
    try {
        cluster_count = stoi(argv[3]);
        if (cluster_count <= 0) throw invalid_argument("non-positive");
    } catch (const exception &) {
        spdlog::error("Invalid cluster count: {}", argv[3]);
        return 1;
    }

    const vector<vector<double> > content = load_csv_dataset(input_filename);
    if (content.empty()) {
        spdlog::error("Input dataset is empty or failed to load: {}", input_filename);
        return 1;
    }

    KMeansClassifier kmeans(content, cluster_count);
    const vector<int> labels = kmeans.fit();

    if (rank == 0) {
        save_csv_dataset(output_filename, content, labels);

        // Compute silhouette score for the clustering and log it.
        try {
            double sil = silhouette::complete_serial(content, labels, cluster_count);
            spdlog::info("Silhouette score: {:.6f}", sil);
            // Warn if silhouette score is low (threshold 0.25 chosen as a heuristic)
            if (sil < 0.25) {
                spdlog::warn("Low silhouette score ({:.6f}). Consider revising the number of clusters, initialization, or preprocessing (scaling/feature selection).", sil);
            }
        } catch (const exception &e) {
            spdlog::error("Failed to compute silhouette score: {}", e.what());
        }
    }

    MPI_Finalize();

    return 0;
}
