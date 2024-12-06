/**
 * @file testbench.cpp
 * @brief 
 * @version 0.1
 * @date 2024-12-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>
#include <fstream>
#include "testbench.h"

using namespace std;

void Testbench::writeImage() {
    ifstream file(img_path);
    string line;

    getline(file, line);            // First line is the label
    getline(file, line);            // Second line is header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int r, g, b;

        char comma;                 // Skip comma between values
        ss >> r >> comma >> g >> comma >> b;

        Pixel pixel;
        pixel.r = static_cast<uint8_t>(r);
        pixel.g = static_cast<uint8_t>(g);
        pixel.b = static_cast<uint8_t>(b);

        // Push the pixel into the FIFO
        fifo.write(pixel);
    }

    file.close();
}