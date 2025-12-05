#include <DataFrame/DataFrame.h>
#include <iostream>
#include <fstream>

using namespace hmdf;
using namespace std;

int load_iris_dataset() {
    StdDataFrame<string> iris_dataset;

    // Load CSV into DataFrame
    iris_dataset.read("dataset.csv", io_format::csv2);
    cout << "Iris Dataset Loaded Successfully!" << endl;

    // cout << "DataFrame Info:" << endl;
    //  description_dataframe = iris_dataset.describe<>();

    const auto  &cool_col_ref = iris_dataset.get_column<string>("species");
    const auto  &str_col_ref = iris_dataset.get_column<string>("species");

    cout << cool_col_ref[1] << cool_col_ref[2] << cool_col_ref[3] << endl;
    cout << "Str Column = ";
    for (const auto &str : str_col_ref)
        cout << str << ", ";
    cout << endl;

    cout << "There are " << iris_dataset.get_column<double>("sepal_width").size()
              << " IBM close prices" << endl;
    cout << "There are " << iris_dataset.get_index().size() << " IBM indices" << endl;


    return 0;
}
