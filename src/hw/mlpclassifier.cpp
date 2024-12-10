/**
 * @file testbench.cpp
 * @brief
 * @version 0.1
 * @date 2024-12-03
 *
 * @copyright Copyright (c) 2024
 *
 */
#define SC_INCLUDE_FX
#include <systemc.h>
#include <sysc/datatypes/fx/sc_fixed.h>
#include "mlpclassifier.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace sc_dt;

/****************************************************************************************
 ********************************** MLP CLASSIFIER **************************************
 ****************************************************************************************/

// Helper function to load 2D CSV files
std::vector<std::vector<float>> loadCSV_2D(const std::string &filepath, int rows, int cols)
{
    std::vector<std::vector<float>> data(rows, std::vector<float>(cols, 0.0f));
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "[ERROR] Unable to open file: " << filepath << std::endl;
        return data;
    }
    std::string line, value;
    int row = 0;
    while (std::getline(file, line) && row < rows)
    {
        std::stringstream ss(line);
        int col = 0;
        while (std::getline(ss, value, ',') && col < cols)
        {
            data[row][col] = std::stof(value);
            col++;
        }
        row++;
    }
    file.close();
    return data;
}

// Helper function to load 1D CSV files
std::vector<float> loadCSV_1D(const std::string &filepath, int size)
{
    std::vector<float> data(size, 0.0f);
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "[ERROR] Unable to open file: " << filepath << std::endl;
        return data;
    }
    std::string line, value;
    int idx = 0;
    while (std::getline(file, line) && idx < size)
    {
        std::stringstream ss(line);
        while (std::getline(ss, value, ',') && idx < size)
        {
            data[idx++] = std::stof(value);
        }
    }
    file.close();
    return data;
}

// Load weights and biases from CSV files
void MLPClassifier::load_weights()
{
    std::string base_dir = "weights/mlp/";

    // Load W1
    W1 = loadCSV_2D(base_dir + "W1.csv", HIDDEN_DIM, EMBED_DIM);
    // Load b1
    b1 = loadCSV_1D(base_dir + "b1.csv", HIDDEN_DIM);
    // Load W2
    W2 = loadCSV_2D(base_dir + "W2.csv", NUM_CLASSES, HIDDEN_DIM);
    // Load b2
    b2 = loadCSV_1D(base_dir + "b2.csv", NUM_CLASSES);

    std::cout << "[INFO] MLPClassifier weights loaded successfully." << std::endl;
}

// Perform the first linear transformation: z = W1 * x + b1
void MLPClassifier::linear_layer1(std::vector<float> &output, const std::vector<float> &input)
{
    for (int i = 0; i < HIDDEN_DIM; ++i)
    {
        float sum = 0.0f;
        for (int j = 0; j < EMBED_DIM; ++j)
        {
            sum += W1[i][j] * input[j];
        }
        sum += b1[i];
        output[i] = sum;
    }
}

// Apply GELU activation: a = GELU(z)
void MLPClassifier::activation_gelu(std::vector<float> &input_output)
{
    for (auto &x : input_output)
    {
        x = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3))));
    }
}

// Perform the second linear transformation: y = W2 * a + b2
void MLPClassifier::linear_layer2(std::vector<float> &output, const std::vector<float> &input)
{
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        float sum = 0.0f;
        for (int j = 0; j < HIDDEN_DIM; ++j)
        {
            sum += W2[i][j] * input[j];
        }
        sum += b2[i];
        output[i] = sum;
    }
}

// Main run method
void MLPClassifier::run()
{
    while (true)
    {
        wait(); // Wait for clock edge

        if (reset.read())
        {
            done.write(false);
            // Optionally, reset internal states if necessary
            continue;
        }

        if (start.read())
        {
            // Load weights (assuming weights are static; consider loading them once during initialization)
            load_weights();

            // Step 1: Input Extraction (Extract [CLS] token)
            std::vector<float> cls_input(EMBED_DIM, 0.0f);
            for (int i = 0; i < EMBED_DIM; ++i)
            {
                cls_input[i] = cls_token[i].read().to_float();
            }

            // Step 2: First Linear Transformation
            std::vector<float> z(HIDDEN_DIM, 0.0f);
            linear_layer1(z, cls_input);

            // Step 3: Activation Function (GELU)
            activation_gelu(z);

            // Step 4: Output Layer
            std::vector<float> y(NUM_CLASSES, 0.0f);
            linear_layer2(y, z);

            // Write output logits
            for (int i = 0; i < NUM_CLASSES; ++i)
            {
                classification[i].write(y[i]);
            }

            // Signal completion
            done.write(true);
        }
    }
}
