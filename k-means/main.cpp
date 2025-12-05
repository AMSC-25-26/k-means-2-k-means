#include <vector>
#include "data_io.cpp"

int main() {
    vector<vector<double> > content = load_csv_dataset("dataset.csv");

    // Print out content
    for (const auto &row: content) {
        for (const auto &val: row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
