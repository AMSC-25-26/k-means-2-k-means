#include <fstream>
#include <vector>
#include <spdlog/spdlog.h>
#include "rapidcsv.h"

using namespace std;
using namespace rapidcsv;

vector<vector<double> > load_csv_dataset(const char *filename) {
    // Loop over each line of the CSV file, and then parse the values for each column. Insert into the resulting matrix only the numeric ones
    Document csv(filename);
    spdlog::info("Loaded CSV with {} rows and {} columns", csv.GetRowCount(), csv.GetColumnCount());

    vector<vector<double> > data;
    // For each row, insert into data
    for (size_t r = 0; r < csv.GetRowCount(); r++) {
        vector<double> row_data;

        for (size_t c = 0; c < csv.GetColumnCount(); c++) {
            try {
                auto val = csv.GetCell<double>(c, r);
                row_data.push_back(val);
            } catch (const std::exception &e) {
                // Non-numeric value, skip
                row_data.push_back(0);
            }
        }

        data.push_back(row_data);
    }

    // If a column is all zeros, remove it
    if (!data.empty()) {
        for (size_t c = 0; c < data[0].size();) {
            bool all_zeros = true;
            for (size_t r = 0; r < data.size(); r++) {
                if (data[r][c] != 0) {
                    all_zeros = false;
                    break;
                }
            }
            if (all_zeros) {
                // Remove column c
                for (size_t r = 0; r < data.size(); r++) {
                    data[r].erase(data[r].begin() + c);
                }
            } else {
                c++;
            }
        }
    }

    return data;
}

int save_csv_dataset(const char *filename, const vector<vector<double> > &data, const vector<int> &labels) {
    ofstream file(filename);
    if (!file.is_open()) {
        spdlog::error("Error opening file for writing: {}", filename);
        return -1;
    }

    Document csv;
    // Write data
    for (size_t r = 0; r < data.size(); r++) {
        for (size_t c = 0; c < data[r].size(); c++) {
            csv.SetCell(c, r, data[r][c]);
        }
        // Write label in the last column
        csv.SetCell(data[r].size(), r, labels[r]);
    }

    csv.Save(file);

    return 0;
}
