/**
 * @file top.cpp
 * @brief Vision transformer
 * @version 0.1
 * @date 2024-12-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <systemc>
#include "top.h"

void VisionTransformer::setParameters() {
    pe.img_path = img_path;
    pe.weights_dir = weights_dir;
    pe.output_buffer = &pe_output_buffer;
    pe.patch_height = patch_size;
    pe.embed_dim = embed_dim;

    tr1.weights_dir = weights_dir;
    tr1.embed_dim = embed_dim;
    tr1.input_buffer = &pe_output_buffer;
    tr1.output_buffer = &tr1_output_buffer;
}

void VisionTransformer::run() {
    // Run the patch embedding layer
    while(true) {
        std::cout << "Weight directory: " << weights_dir << std::endl;
        pe_start = 1;
        wait(pe_done.posedge_event());
        pe_start = 0;

        wait(10, SC_NS);
        // std::cout << pe_output_buffer << std::endl;
        // Run the transformer
        tr1_start = 1;
        wait(tr1_done.posedge_event());
        tr1_start = 0;

        // Run the fully connected layer
        // fc.run();

        // Output the classification
        classification.write(10);
    }
}