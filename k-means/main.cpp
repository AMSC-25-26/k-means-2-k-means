#include <iostream>
#include <vector>
#include <string>
#include <spdlog/spdlog.h>
#include "data_io.cpp"
#include "KMeansClassifier.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        spdlog::error("Usage: {} <input_csv> <output_csv> <cluster_count>", argv[0]);
        return 1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    int cluster_count = 0;
    try {
        cluster_count = std::stoi(argv[3]);
        if (cluster_count <= 0) throw std::invalid_argument("non-positive");
    } catch (const std::exception&) {
        spdlog::error("Invalid cluster count: {}", argv[3]);
        return 1;
    }

    std::vector<std::vector<double>> content = load_csv_dataset(input_filename);
    KMeansClassifier kmeans(content, cluster_count);
    std::vector<int> labels = kmeans.fit();

    save_csv_dataset(output_filename, content, labels);

    return 0;
}
