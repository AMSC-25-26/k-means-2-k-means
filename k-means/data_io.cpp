#include <fstream>
#include <iostream>
#include <vector>
#include "rapidcsv.h"

using namespace std;
using namespace rapidcsv;

vector<vector<double> > load_csv_dataset(const char *filename) {
    // Loop over each line of the CSV file, and then parse the values for each column. Insert into the resulting matrix only the numeric ones
    Document csv(filename);
    printf("Loaded CSV with %lu rows and %lu columns", csv.GetRowCount(), csv.GetColumnCount());

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
