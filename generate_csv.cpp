//
// Created by Avi Kapadia on 11/17/24.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
using namespace std;

void generateMassiveCSV(const string &filename, int numRows, int startID) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }
    file << "ID,Value\n";
    for (int i = 0; i < numRows; ++i) {
        file << startID + i << ",Value" << (startID + i) << "\n";
    }
    file.close();
}

// Generate massive CSV files for testing
int main() {
    int numRows = 10000000;
    // Generate dynamic filenames
    std::string file1 = "massive_table" + std::to_string(numRows) + "-1.csv";
    std::string file2 = "massive_table" + std::to_string(numRows) + "-2.csv";

    generateMassiveCSV(file1, numRows, 1);  // IDs 1 to 1,000,000
    generateMassiveCSV(file2, numRows, numRows / 2);  // Overlapping IDs: 500,000 to 1,500,000
}