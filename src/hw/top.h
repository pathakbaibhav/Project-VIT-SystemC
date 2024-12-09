/**
 * @file top.cpp
 * @brief Top level module for the Vision Transformer chip
 * @version 0.1
 * @date 2024-11-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <systemc>
#include "patchEmbedding.h"

using namespace sc_core;

SC_MODULE (VisionTransformer) {
    public:
        /** Input/Output */
        std::string img_path;           // Path to the input image
        std::string weights_dir;        // Directory with all of the weight files
        int patch_size;
        int embed_dim;

        sc_in<int> img_length;          // Will always be 256
        sc_in<int> img_height;          // Will always be 256
        sc_in<int> img_channels;        // Will always be 3
        sc_in<bool> start;              // Start signal, in place of image fifo for now
        sc_out<int> classification;     // Final classification from the model

        /** Internal signals/values */
        float* pe_output_buffer;        // Array to store the output of the patch embedding

        sc_signal<bool> pe_start;       // Signals to start the patch embedding
        sc_signal<bool> pe_done;        // Signals that the patch embedding is done

        /** Layers */
        PatchEmbedding pe;
        // transformer tr;
        // fullyConnected fc;

        void run();                     // Run inference

        SC_CTOR(VisionTransformer) : pe("pe") {
            // c++
            pe.img_path = img_path;
            pe.weights_dir = weights_dir;
            pe.output_buffer = pe_output_buffer;
            pe.patch_size = patch_size;
            pe.embed_dim = embed_dim;

            // systemc
            pe.img_length(img_length);
            pe.img_height(img_height);
            pe.img_channels(img_channels);
            pe.start(pe_start);
            pe.done(pe_done);

            SC_THREAD(run);             // Eventually this should be sensitive to the image being completely loaded
        }
};