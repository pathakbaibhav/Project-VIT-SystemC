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
#include <top.h>

void VisionTransformer::run() {
    pe_output_buffer = new float[embed_dim];    // Allocate array to store output of Patch Embedding layer
    
    pe_done = 0;
}