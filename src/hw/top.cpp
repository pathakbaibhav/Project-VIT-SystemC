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

void VisionTransformer::run() {
    // Run the patch embedding layer
    while(true) {
        pe_start = 1;
        pe_output_buffer = new float[embed_dim];    // Allocate array to store output of Patch Embedding layer    
        wait(pe_done.posedge_event());
        pe_start = 0;

        // Run the transformer
        // tr.run();

        // Run the fully connected layer
        // fc.run();

        // Output the classification
        classification.write(10);

        free(pe_output_buffer);     // Free the memory allocated for the output buffer
    }
}