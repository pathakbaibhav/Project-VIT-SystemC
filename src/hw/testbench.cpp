/**
 * @file testbench.cpp
 * @brief 
 * @version 0.1
 * @date 2024-12-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <systemc.h>
#include <iostream>
#include <string>
#include <fstream>
#include "testbench.h"

using namespace std;

void Testbench::setParameters() {
    vt.img_path = img_path;
    vt.weights_dir = weights_dir;
    vt.patch_size = patch_size;
    vt.embed_dim = embed_dim;

    vt.setParameters();
}

void Testbench::stim() {                // SC_THREAD
    img_length = IMAGE_WIDTH;
    img_height = IMAGE_HEIGHT;
    img_channels = IMAGE_CHANNELS;

    start_inference = 1;          // Run inference
    wait(10, SC_NS);
}

void Testbench::getClassification() {   // SC_METHOD
    cout << "Image path: " << img_path << "  ****  ";
    cout << "Weight directory: " << weights_dir << endl;
    cout << "Classification: " << classification << endl;
}

int sc_main(int argc, char *argv[])
{
    // Instantiate your top-level module
    VisionTransformer vt("vision_transformer");
    // TopModule top("top");
    // Set up any necessary signals or stimuli here

    // Run the simulation
    sc_start();

    return 0;
}