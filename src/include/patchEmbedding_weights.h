#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Dense>
#include <vector>

/**
 * @brief Get the Weights from a CSV file.
 * 
 * Reads a CSV file and returns an Eigen matrix with the specified dimensions.
 * 
 * @param file_path Path to the CSV file.
 * @param dim1 Number of rows in the resulting matrix.
 * @param dim2 Number of columns in the resulting matrix.
 * @return Eigen::MatrixXf The resulting matrix.
 */
Eigen::MatrixXf getWeights(const std::string& file_path, const int dim1, const int dim2) {
    // Open the CSV file
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + file_path);
    }

    std::string line;
    std::vector<float> values;

    // Read each line and parse values
    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        std::string value;

        while (std::getline(line_stream, value, ',')) {
            try {
                values.push_back(std::stof(value));  // Convert string to float and add to vector
            } catch (const std::exception& e) {
                throw std::runtime_error("Error parsing value: " + value + " in file: " + file_path);
            }
        }
    }

    // Close the file
    file.close();

    // Validate dimensions
    if (values.size() != static_cast<size_t>(dim1 * dim2)) {
        throw std::runtime_error("Mismatch between provided dimensions and number of values in file: " + file_path);
    }

    // Create Eigen Matrix
    Eigen::MatrixXf matrix(dim1, dim2);

    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            matrix(i, j) = values[i * dim2 + j];  // Fill the matrix with values from the CSV
        }
    }

    return matrix;
}

/**
 * @brief Get the Weights from a CSV file as a VectorXf.
 * 
 * Reads a CSV file and returns an Eigen vector with the specified length.
 * 
 * @param file_path Path to the CSV file.
 * @param dim Length of the resulting vector.
 * @return Eigen::VectorXf The resulting vector.
 */
Eigen::VectorXf getWeights(const std::string& file_path, const int dim) {
    // Open the CSV file
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + file_path);
    }

    std::string line;
    std::vector<float> values;

    // Read each line and parse values
    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        std::string value;

        while (std::getline(line_stream, value, ',')) {
            try {
                values.push_back(std::stof(value));  // Convert string to float and add to vector
            } catch (const std::exception& e) {
                throw std::runtime_error("Error parsing value: " + value + " in file: " + file_path);
            }
        }
    }

    // Close the file
    file.close();

    // Validate dimensions
    if (values.size() != static_cast<size_t>(dim)) {
        throw std::runtime_error("Mismatch between provided length and number of values in file: " + file_path);
    }

    // Create Eigen Vector
    Eigen::VectorXf vector(dim);
    for (int i = 0; i < dim; ++i) {
        vector(i) = values[i];
    }

    return vector;
}
