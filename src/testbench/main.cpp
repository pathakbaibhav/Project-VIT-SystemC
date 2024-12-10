/**
 * @file main.cpp
 * @brief 
 * @version 0.1
 * @date 2024-12-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <systemc>
#include <iostream>
#include <string>
#include "../hw/testbench.h"

using namespace std;
using namespace sc_core;

int sc_main(int argc, char* argv[]) 
{
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <img_path> <weights_path>" << endl;
        return 1;
    }

    Testbench tb("tb");

    tb.img_path = argv[1];
    tb.weights_dir = argv[2];
    tb.patch_size = 16;
    tb.embed_dim = 768;

    tb.setParameters();

    cout << "Starting simulation" << endl;
    sc_start();

    cout << "Simulation finished" << endl;

    return 0;
}
