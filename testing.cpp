//
// Created by Avi Kapadia on 11/4/24.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace std;

// this function will create a sample csv file

void createCSV(const string &filename, const vector<vector<string>> &data) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }
    for (const auto &row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

// this function will read a csv file and return a map for joining
vector<vector<string>> readCSV(const string &filename) {
    vector<vector<string>> table;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return table;
    }

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<string> row;

        // Split the line by commas and push values into the row
        while (getline(ss, value, ',')) {
            row.push_back(value);
        }

        // Add the row to the table
        table.push_back(row);
    }

    file.close();
    return table;
}

// this function will perform inner join WITHOUT cuda

const unordered_map<string, vector<string>>;

void innerJoin(const vector<vector<string>> &table1,
               const vector<vector<string>> &table2,
               const string &column1,
               const string &column2) {
    // Find the indexes of the columns to join on
    int colIndex1 = -1, colIndex2 = -1;

    // Search for column1 in the header of table1
    for (int i = 0; i < table1[0].size(); ++i) {
        if (table1[0][i] == column1) {
            colIndex1 = i;
            break;
        }
    }

    // Search for column2 in the header of table2
    for (int i = 0; i < table2[0].size(); ++i) {
        if (table2[0][i] == column2) {
            colIndex2 = i;
            break;
        }
    }

    // If either column is not found, return an error message
    if (colIndex1 == -1 || colIndex2 == -1) {
        cout << "Unable to perform join: attribute not found\n";
        return;
    }

    // Print the header for the joined table
    cout << "Joined Table (Inner Join):\n";
    for (const string &col : table1[0]) cout << col << ",";
    for (int i = 0; i < table2[0].size(); ++i) {
        if (i != colIndex2) cout << table2[0][i] << ",";
    }
    cout << "\n";

    // Perform the join
    for (size_t i = 1; i < table1.size(); ++i) {
        for (size_t j = 1; j < table2.size(); ++j) {
            if (table1[i][colIndex1] == table2[j][colIndex2]) {
                // Print the matching row
                for (const string &val : table1[i]) cout << val << ",";
                for (size_t k = 0; k < table2[j].size(); ++k) {
                    if (k != colIndex2) cout << table2[j][k] << ",";
                }
                cout << "\n";
            }
        }
    }
}


// this function will perform inner join WITH cuda

//TODO: Implement innerJoinCuda

// main
//print function
void printTable(const vector<vector<string>> &data2) {
    for (const auto &row : data2) {
        for (const auto &val : row) cout << val << ",";
        cout << "\n";
    }
    cout << "\n";
}

int main() {
    // create two sample csv files and perform inner join



    // test 1: simple join -------------------------

    // Sample data for CSV files (normally i will download these from somewhere, so right now it seems redundant)
    vector<vector<string>> data1 = {{"ID", "Name"}, {"1", "Alice"}, {"2", "Bob"}, {"3", "Charlie"}};
    vector<vector<string>> data2 = {{"ID", "Age"}, {"1", "23"}, {"2", "34"}, {"4", "45"}};

    // print the oriiinal data before join
    std::cout << "Table 1:\n";
    printTable(data1);
    std::cout << "Table 2:\n";
    printTable(data2);

    // create CSV files
    createCSV("table1.csv", data1);
    createCSV("table2.csv", data2);

    // read CSV files into maps
    auto table1 = readCSV("table1.csv");
    auto table2 = readCSV("table2.csv");

    // perform inner join
    innerJoin(table1, table2, "ID", "ID");

    // test 2: simple join with no matching keys -------------------------
    vector<vector<string>> data3 = {{"ID", "Name"}, {"1", "Alice"}, {"2", "Bob"}, {"3", "Charlie"}};
    vector<vector<string>> data4 = {{"ID", "Age"}, {"4", "23"}, {"5", "34"}, {"6", "45"}};

    // print the oriiinal data before join
    std::cout << "Table 3:\n";
    printTable(data3);
    std::cout << "Table 4:\n";
    printTable(data4);

    // create CSV files
    createCSV("table3.csv", data3);
    createCSV("table4.csv", data4);

    auto table3 = readCSV("table3.csv");
    auto table4 = readCSV("table4.csv");
    innerJoin(table3, table4, "ID", "ID");

    // test 3: simple join with attributes of different name -------------------------
    vector<vector<string>> data5 = {{"ID", "Name"}, {"1", "Alice"}, {"2", "Bob"}, {"3", "Charlie"}};
    vector<vector<string>> data6 = {{"Real_ID", "Age"}, {"1", "23"}, {"2", "34"}, {"3", "45"}};

    //print
    std::cout << "Table 5:\n";
    printTable(data5);
    std::cout << "Table 6:\n";
    printTable(data6);
    createCSV("table5.csv", data5);
    createCSV("table6.csv", data6);
    auto table5 = readCSV("table5.csv");
    auto table6 = readCSV("table6.csv");
    // should return cartesian product
    innerJoin(table5, table6, "ID", "Real_ID");


    return 0;
}