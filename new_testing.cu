//
// Created by Avi Kapadia on 11/4/24.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>

// for cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

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
               const string &column2,
               const string &outputFilename) {


    ofstream outFile(outputFilename);
    if (!outFile.is_open()) {
        cerr << "Failed to open output file: " << outputFilename << endl;
        return;
    }

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
        cerr << "Unable to perform join: attribute not found\n";
        outFile.close();
        return;
    }

    // Print the header for the joined table
//    cout << "Joined Table (Inner Join):\n";
    for (const string &col : table1[0]) outFile << col << ",";
    for (int i = 0; i < table2[0].size(); ++i) {
        if (i != colIndex2) outFile << table2[0][i] << ",";
    }
    outFile << "\n";

    // Perform the join
    for (size_t i = 1; i < table1.size(); ++i) {
        for (size_t j = 1; j < table2.size(); ++j) {
            if (table1[i][colIndex1] == table2[j][colIndex2]) {
                // Print the matching row
                for (const string &val : table1[i]) outFile << val << ",";
                for (size_t k = 0; k < table2[j].size(); ++k) {
                    if (k != colIndex2) outFile << table2[j][k] << ",";
                }
                outFile << "\n";
            }
        }
    }
    outFile << "\n";
    outFile.close();
}


// this function will perform inner join WITH cuda

//TODO: Implement innerJoinCuda

// first cuda kernel

__global__ void innerJoinKernel(const char *table1Keys, const char *table2Keys,
                                int *matches, int table1Size, int table2Size, int keyLength) {
    int tid1 = blockIdx.x * blockDim.x + threadIdx.x;
    int tid2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (tid1 < table1Size && tid2 < table2Size) {
        // Compare the keys
        bool isMatch = true;
        for (int i = 0; i < keyLength; i++) {
            if (table1Keys[tid1 * keyLength + i] != table2Keys[tid2 * keyLength + i]) {
                isMatch = false;
                break;
            }
        }

        // If match, record the indices
        if (isMatch) {
            atomicAdd(&matches[tid1 * table2Size + tid2], 1);  // Use atomic operation for thread safety
        }
    }
}


void cudaInnerJoin(const vector<vector<string>> &table1,
                   const vector<vector<string>> &table2,
                   const string &column1, const string &column2,
                   const string &outputFilename) {

    ofstream outFile(outputFilename);
    if (!outFile.is_open()) {
        cerr << "Failed to open output file: " << outputFilename << endl;
        return;
    }

    // Find column indices
    int colIndex1 = -1, colIndex2 = -1;
    for (int i = 0; i < table1[0].size(); ++i) {
        if (table1[0][i] == column1) colIndex1 = i;
    }
    for (int i = 0; i < table2[0].size(); ++i) {
        if (table2[0][i] == column2) colIndex2 = i;
    }

    if (colIndex1 == -1 || colIndex2 == -1) {
        cerr << "Unable to perform join: attribute not found\n";
        outFile.close();
        return;
    }

    // Prepare data
    vector<string> table1Keys, table2Keys;
    for (size_t i = 1; i < table1.size(); ++i) {
        table1Keys.push_back(table1[i][colIndex1]);
    }
    for (size_t i = 1; i < table2.size(); ++i) {
        table2Keys.push_back(table2[i][colIndex2]);
    }

    // Flatten keys into a single buffer
    int keyLength = table1Keys[0].size();  // Assume fixed-length keys
    char *h_table1Keys = new char[table1Keys.size() * keyLength];
    char *h_table2Keys = new char[table2Keys.size() * keyLength];
    for (size_t i = 0; i < table1Keys.size(); ++i) {
        memcpy(h_table1Keys + i * keyLength, table1Keys[i].c_str(), keyLength);
    }
    for (size_t i = 0; i < table2Keys.size(); ++i) {
        memcpy(h_table2Keys + i * keyLength, table2Keys[i].c_str(), keyLength);
    }

    // Allocate device memory for table2
    char *d_table2Keys;
    cudaMalloc(&d_table2Keys, table2Keys.size() * keyLength);
    cudaMemcpy(d_table2Keys, h_table2Keys, table2Keys.size() * keyLength, cudaMemcpyHostToDevice);

    int *d_matches;
    int table1Size = table1Keys.size(), table2Size = table2Keys.size();

    // **Batching logic for table1**
    int batchSize = 10000;  // Adjust this based on available GPU memory
    int *h_matches = new int[batchSize * table2Size];

    for (int batchStart = 0; batchStart < table1Size; batchStart += batchSize) {
        int currentBatchSize = min(batchSize, table1Size - batchStart);

        // Allocate memory for current batch
        char *d_table1Keys;
        cudaMalloc(&d_table1Keys, currentBatchSize * keyLength);
        cudaMemcpy(d_table1Keys, h_table1Keys + batchStart * keyLength, currentBatchSize * keyLength, cudaMemcpyHostToDevice);

        cudaMalloc(&d_matches, currentBatchSize * table2Size * sizeof(int));
        cudaMemset(d_matches, 0, currentBatchSize * table2Size * sizeof(int));

        // Launch kernel for this batch
        dim3 blockSize(16, 16);
        dim3 gridSize((currentBatchSize + blockSize.x - 1) / blockSize.x,
                      (table2Size + blockSize.y - 1) / blockSize.y);
        innerJoinKernel<<<gridSize, blockSize>>>(d_table1Keys, d_table2Keys, d_matches,
                                                 currentBatchSize, table2Size, keyLength);

        // Copy results back to host
        cudaMemcpy(h_matches, d_matches, currentBatchSize * table2Size * sizeof(int), cudaMemcpyDeviceToHost);

        // Write matches to the output file
        for (int i = 0; i < currentBatchSize; ++i) {
            for (int j = 0; j < table2Size; ++j) {
                if (h_matches[i * table2Size + j] > 0) {
                    // Write the matching row to the file
                    cout << "Match found: " << table1[batchStart + i + 1][colIndex1] << " - " << table2[j + 1][colIndex2] << "\n";
                    for (const string &val : table1[batchStart + i + 1]) outFile << val << ",";
                    for (size_t k = 0; k < table2[j + 1].size(); ++k) {
                        if (k != colIndex2) outFile << table2[j + 1][k] << ",";
                    }
                    outFile << "\n";
                }
            }
        }

        // Cleanup for current batch
        cudaFree(d_table1Keys);
        cudaFree(d_matches);
    }

    // Cleanup
    outFile.close();
    cudaFree(d_table2Keys);
    delete[] h_table1Keys;
    delete[] h_table2Keys;
    delete[] h_matches;
}


// main
//print function
void printTable(const vector<vector<string>> &data2) {
    for (const auto &row : data2) {
        for (const auto &val : row) cout << val << ",";
        cout << "\n";
    }
    cout << "\n";
}

// use this function to time cpu vs gpu time

template <typename Func>
double measureExecutionTime(Func functionToTime) {
    auto start = std::chrono::high_resolution_clock::now();
    functionToTime();
    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

// using this function to generate a massive CSV file for testing

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
    innerJoin(table1, table2, "ID", "ID", "output_test1.csv");

    // print the joined data
    auto joinedTable = readCSV("output_test1.csv");

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
    innerJoin(table3, table4, "ID", "ID", "output_test2.csv");

    // print the joined data
    auto joinedTable2 = readCSV("output_test2.csv");

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
    innerJoin(table5, table6, "ID", "Real_ID", "output_test3.csv");

    // print the joined data
    auto joinedTable3 = readCSV("output_test3.csv");

    // test 4 --- join with CPU vs GPU size 100

    std::cout << "Test 4: Inner Join with CPU vs GPU size 100\n";
    // Read the massive CSV files into memory
    auto table7 = readCSV("massive_tables/massive_table100-1.csv");
    auto table8 = readCSV("massive_tables/massive_table100-2.csv");

    // Time the CPU join
    std::cout << "Massive Inner Join: beginning \n";

    double cpuTime = measureExecutionTime([&]() {
        innerJoin(table7, table8, "ID", "ID", "output_test4_cpu.csv");
    });
    std::cout << "CPU Inner Join Time: " << cpuTime << " seconds\n";

    std::cout << "Massive Inner Join: finished \n";
    // Time the CUDA join

    std::cout << "Massive CUDA Inner Join: beginning \n";

    double cudaTime = measureExecutionTime([&]() {
        cudaInnerJoin(table7, table8, "ID", "ID", "output_test4_cuda.csv");
    });
    std::cout << "CUDA Inner Join Time: " << cudaTime << " seconds\n";

    std::cout << "Massive CUDA Inner Join: finished \n";

    // print GPU speed up factor
    std::cout << "GPU Speedup Factor: " << cpuTime / cudaTime << "\n";

    // test 5 --- join with CPU vs GPU size 1000 -------------------------
    std::cout << "Test 5: Inner Join with CPU vs GPU size 1000\n";

    // Read the massive CSV files into memory
    table7 = readCSV("massive_tables/massive_table1000-1.csv");
    table8 = readCSV("massive_tables/massive_table1000-2.csv");

    // Time the CPU join
    std::cout << "Massive Inner Join: beginning \n";

    cpuTime = measureExecutionTime([&]() {
        innerJoin(table7, table8, "ID", "ID", "output_test5_cpu.csv");
    });
    std::cout << "CPU Inner Join Time: " << cpuTime << " seconds\n";

    std::cout << "Massive Inner Join: finished \n";
    // Time the CUDA join

    std::cout << "Massive CUDA Inner Join: beginning \n";

    cudaTime = measureExecutionTime([&]() {
        cudaInnerJoin(table7, table8, "ID", "ID", "output_test5_cuda.csv");
    });
    std::cout << "CUDA Inner Join Time: " << cudaTime << " seconds\n";

    std::cout << "Massive CUDA Inner Join: finished \n";

    // print GPU speed up factor
    std::cout << "GPU Speedup Factor: " << cpuTime / cudaTime << "\n";

    // test 6 --- join with CPU vs GPU size 10000 -------------------------
    std::cout << "Test 6: Inner Join with CPU vs GPU size 10000\n";

    // Read the massive CSV files into memory
    table7 = readCSV("massive_tables/massive_table10000-1.csv");
    table8 = readCSV("massive_tables/massive_table10000-2.csv");

    // Time the CPU join
    std::cout << "Massive Inner Join: beginning \n";

    cpuTime = measureExecutionTime([&]() {
        innerJoin(table7, table8, "ID", "ID", "output_test6_cpu.csv");
    });
    std::cout << "CPU Inner Join Time: " << cpuTime << " seconds\n";

    std::cout << "Massive Inner Join: finished \n";
    // Time the CUDA join

    std::cout << "Massive CUDA Inner Join: beginning \n";

    cudaTime = measureExecutionTime([&]() {
        cudaInnerJoin(table7, table8, "ID", "ID", "output_test6_cuda.csv");
    });
    std::cout << "CUDA Inner Join Time: " << cudaTime << " seconds\n";

    std::cout << "Massive CUDA Inner Join: finished \n";

    // print GPU speed up factor
    std::cout << "GPU Speedup Factor: " << cpuTime / cudaTime << "\n";

    // test 7 --- join with CPU vs GPU size 100000 -------------------------
    std::cout << "Test 7: Inner Join with CPU vs GPU size 100000\n";

    // Read the massive CSV files into memory
    table7 = readCSV("massive_tables/massive_table100000-1.csv");
    table8 = readCSV("massive_tables/massive_table100000-2.csv");

    // Time the CPU join
    std::cout << "Massive Inner Join: beginning \n";

    cpuTime = measureExecutionTime([&]() {
        innerJoin(table7, table8, "ID", "ID", "output_test7_cpu.csv");
    });
    std::cout << "CPU Inner Join Time: " << cpuTime << " seconds\n";

    std::cout << "Massive Inner Join: finished \n";
    // Time the CUDA join

    std::cout << "Massive CUDA Inner Join: beginning \n";

    cudaTime = measureExecutionTime([&]() {
        cudaInnerJoin(table7, table8, "ID", "ID", "output_test7_cuda.csv");
    });
    std::cout << "CUDA Inner Join Time: " << cudaTime << " seconds\n";

    std::cout << "Massive CUDA Inner Join: finished \n";

    // print GPU speed up factor
    std::cout << "GPU Speedup Factor: " << cpuTime / cudaTime << "\n";

    // test 8 --- join with CPU vs GPU size 1000000 -------------------------
    std::cout << "Test 8: Inner Join with CPU vs GPU size 1000000\n";

    // Read the massive CSV files into memory
    table7 = readCSV("massive_tables/massive_table1000000-1.csv");
    table8 = readCSV("massive_tables/massive_table1000000-2.csv");

    // Time the CPU join
    std::cout << "Massive Inner Join: beginning \n";

    cpuTime = measureExecutionTime([&]() {
        innerJoin(table7, table8, "ID", "ID", "output_test8_cpu.csv");
    });
    std::cout << "CPU Inner Join Time: " << cpuTime << " seconds\n";

    std::cout << "Massive Inner Join: finished \n";
    // Time the CUDA join

    std::cout << "Massive CUDA Inner Join: beginning \n";

    cudaTime = measureExecutionTime([&]() {
        cudaInnerJoin(table7, table8, "ID", "ID", "output_test8_cuda.csv");
    });
    std::cout << "CUDA Inner Join Time: " << cudaTime << " seconds\n";

    std::cout << "Massive CUDA Inner Join: finished \n";

    // print GPU speed up factor
    std::cout << "GPU Speedup Factor: " << cpuTime / cudaTime << "\n";

    // test 9 -- just gpu with size 10000000 - ten  million

    std::cout << "Test 9: Inner Join with GPU size 10000000\n";

    table7 = readCSV("massive_tables/massive_table10000000-1.csv");
    table8 = readCSV("massive_tables/massive_table10000000-2.csv");

    std::cout << "Massive CUDA Inner Join: beginning \n";

    cudaTime = measureExecutionTime([&]() {
        cudaInnerJoin(table7, table8, "ID", "ID", "output_test9_cuda.csv");
    });
    std::cout << "CUDA Inner Join Time: " << cudaTime << " seconds\n";

    std::cout << "Massive CUDA Inner Join: finished \n";

    // no gpu speed up factor as we are not comparing with cpu

    return 0;
}