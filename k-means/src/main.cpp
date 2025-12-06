#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <spdlog/spdlog.h>
#include "data_io.cpp"
#include "KMeansClassifier.h"


int main(int argc, char *argv[]) {
    spdlog::set_level(spdlog::level::debug);

    if (argc < 4) {
        spdlog::error("Usage: {} <input_csv> <output_csv> <cluster_count> [true_labels_file]", argv[0]);
        return 1;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];

    int cluster_count = 0;
    try {
        cluster_count = std::stoi(argv[3]);
        if (cluster_count <= 0) throw std::invalid_argument("non-positive");
    } catch (const std::exception &) {
        spdlog::error("Invalid cluster count: {}", argv[3]);
        return 1;
    }

    const std::vector<std::vector<double> > content = load_csv_dataset(input_filename);
    if (content.empty()) {
        spdlog::error("Input dataset is empty or failed to load: {}", input_filename);
        return 1;
    }

    // Optional: load true labels if provided as 4th argument
    std::vector<std::string> true_labels;
    if (argc >= 5) {
        const char *true_labels_file = argv[4];
        true_labels = load_label_file(true_labels_file);

        if (true_labels.empty()) {
            spdlog::error("No labels loaded from {}", true_labels_file);
            return 1;
        }

        if (true_labels.size() != content.size()) {
            spdlog::error("Label count ({}) does not match number of samples ({})", true_labels.size(), content.size());
            return 1;
        }

        spdlog::info("Loaded true labels from {}", true_labels_file);
    }

    KMeansClassifier kmeans(content, cluster_count);
    const std::vector<int> labels = kmeans.fit();

    save_csv_dataset(output_filename, content, labels);

    if (!true_labels.empty()) {
        spdlog::info("Predicted {} labels; true labels provided for {} samples", labels.size(), true_labels.size());
        // Further evaluation (e.g., AMI) can be added here using the loaded true_labels.
    }

    return 0;
}
