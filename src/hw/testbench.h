/**
 * @file testbench.cpp
 * @brief 
 * @version 0.1
 * @date 2024-12-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include <systemc.h>
#include <string>
#include "top.h"

const int IMAGE_WIDTH = 224;
const int IMAGE_HEIGHT = 224;
const int IMAGE_CHANNELS = 3;


SC_MODULE(Testbench) {
     public:
        std::string img_path;               // Path to the image
        std::string weights_dir;            // Directory holding all of the weights
        int patch_size;                     // Patch size - always set to 16 for now
        int embed_dim;                      // Dimension of the embedding layer - set to 724

        // sc_fifo<Pixel> img;                 // Input image, send this pixel by pixel via a fifo
        sc_signal<int> img_length;          // Will always be 256
        sc_signal<int> img_height;          // Will always be 256
        sc_signal<int> img_channels;        // Will always be 3
        sc_signal<bool> start_inference;    // Starts the inference in the vision transformer
        sc_signal<int> classification;      // The classification, outputted as an int

        VisionTransformer vt;               // Call the top module

        // void writeImage();
        void setParameters();
        void stim();
        void getClassification();

        SC_CTOR(Testbench) : vt("vt") {
            // c++
            vt.img_path = img_path;
            vt.weights_dir = weights_dir;
            vt.patch_size = patch_size;
            vt.embed_dim = embed_dim;

            // systemc
            vt.img_length(img_length);
            vt.img_height(img_height);
            vt.img_channels(img_channels);
            vt.start(start_inference);
            vt.classification(classification);

            SC_THREAD(stim);                // Calls the vision transformer
            SC_METHOD(getClassification);   // Gets the output from the vision transformer
            sensitive << classification;    // Sensitive to the classification
        }
};