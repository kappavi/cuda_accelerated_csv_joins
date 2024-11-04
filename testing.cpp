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
unordered_map<string, vector<string>> readCSV(const string &filename) {
    unordered_map<string, vector<string>> dataMap;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return dataMap;
    }

    while (getline(file, line)) {
        stringstream ss(line);
        string key, value;
        vector<string> row;

        getline(ss, key, ',');  // Use the first column as key
        while (getline(ss, value, ',')) {
            row.push_back(value);
        }

        dataMap[key] = row;
    }
    file.close();
    return dataMap;
}

// this function will perform inner join WITHOUT cuda

void innerJoin(const unordered_map<string, vector<string>> &table1,
               const unordered_map<string, vector<string>> &table2) {

    cout << "Joined Table (Inner Join):\n";
    for (const auto &pair : table1) {
        if (table2.find(pair.first) != table2.end()) {
            cout << pair.first;
            for (const auto &val : pair.second) cout << "," << val;
            for (const auto &val : table2.at(pair.first)) cout << "," << val;
            cout << "\n";
        }
    }
}

// this function will perform inner join WITH cuda

//TODO: Implement innerJoinCuda

// main

int main() {
    // create two sample csv files and perform inner join




    // Sample data for CSV files
    vector<vector<string>> data1 = {{"ID", "Name"}, {"1", "Alice"}, {"2", "Bob"}, {"3", "Charlie"}};
    vector<vector<string>> data2 = {{"ID", "Age"}, {"1", "23"}, {"2", "34"}, {"4", "45"}};

    // print the oriiinal data before join

    std::cout << "Table 1:\n";
    for (const auto &row : data1) {
        for (const auto &val : row) cout << val << ",";
        cout << "\n";
    }
    std::cout << "Table 2:\n";
    for (const auto &row : data2) {
        for (const auto &val : row) cout << val << ",";
        cout << "\n";
    }

    // Create CSV files
    createCSV("table1.csv", data1);
    createCSV("table2.csv", data2);

    // Read CSV files into maps
    auto table1 = readCSV("table1.csv");
    auto table2 = readCSV("table2.csv");

    // Perform inner join
    innerJoin(table1, table2);

    return 0;
}