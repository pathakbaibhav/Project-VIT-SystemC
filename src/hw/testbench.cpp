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

void Testbench::stim() {                // SC_THREAD
    img_length = 224;
    img_height = 224;
    img_channel = 3;

    start = 1;          // Run inference
    wait(10, SC_NS);
}

void Testbench::getClassification() {   // SC_METHOD
    cout << "Image path: " << img_path << "  ****  ";
    cout << "Classification: " << classification << endl;
}