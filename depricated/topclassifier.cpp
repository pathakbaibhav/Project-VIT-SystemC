/**
 * @file testbench.cpp
 * @brief
 * @version 0.1
 * @date 2024-12-03
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "top.h"
#include <iostream>

void VisionTransformer::setParameters()
{
    // Implement parameter setting logic here
}

void VisionTransformer::run()
{
    while (true)
    {
        wait(start.posedge_event()); // Wait for the start signal

        // Start Patch Embedding
        pe_start.write(true);
        wait(pe_done.posedge_event());
        pe_start.write(false);

        // Start Transformer
        tr1_start.write(true);
        wait(tr1_done.posedge_event());
        tr1_start.write(false);

        // Implement classification logic here
        // You might need to add a fully connected layer or use the transformer output directly

        // For now, let's just set a dummy classification
        classification.write(0); // Replace with actual classification logic

        std::cout << "Classification complete." << std::endl;
    }
}

// If you need any additional method implementations, add them here