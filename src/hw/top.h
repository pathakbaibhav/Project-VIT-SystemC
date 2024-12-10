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
#include <Eigen/Dense>
#include "patchEmbedding.h"
#include "transformer.h"

using namespace sc_core;

SC_MODULE (VisionTransformer) {
    public:
        /** Input/Output */
        std::string img_path;               // Path to the input image
        std::string weights_dir;            // Directory with all of the weight files
        int patch_size;
        int embed_dim;

        sc_in<int> img_length;              // Will always be 224
        sc_in<int> img_height;              // Will always be 224
        sc_in<int> img_channels;            // Will always be 3
        sc_in<bool> start;                  // Start signal, in place of image fifo for now
        sc_out<int> classification;         // Final classification from the model

        /** Internal signals/values */
        Eigen::MatrixXf pe_output_buffer;   // Matrix to store the output of the patch embedding
        Eigen::MatrixXf tr1_output_buffer;  // Matrix to store the output of the transformer

        sc_signal<bool> pe_start;           // Signals to start the patch embedding
        sc_signal<bool> pe_done;            // Signals that the patch embedding is done

        sc_signal<bool> tr1_start;          // Signals to start the transformer
        sc_signal<bool> tr1_done;           // Signals that the transformer is done

        /** Layers */
        PatchEmbedding pe;
        Transformer tr1;
        // fullyConnected fc;

        void setParameters();               // Set the parameters for the vision transformer
        void run();                         // Run inference

        SC_CTOR(VisionTransformer) : pe("pe"), tr1("tr1") {
            // Patch Embedding
            pe.img_path = img_path;
            pe.weights_dir = weights_dir;
            pe.output_buffer = &pe_output_buffer;   // Note we pass the buffer by reference here, pe is expecting a ptr
            pe.patch_height = patch_size;
            pe.embed_dim = embed_dim;

            pe.img_length(img_length);
            pe.img_height(img_height);
            pe.img_channels(img_channels);
            pe.start(pe_start);
            pe.done(pe_done);

            // Transformer
            tr1.weights_dir = weights_dir;
            tr1.embed_dim = embed_dim;
            tr1.input_buffer = &pe_output_buffer;
            tr1.output_buffer = &tr1_output_buffer;

            tr1.start(tr1_start);
            tr1.done(tr1_done);

            SC_THREAD(run);                 // Eventually this could be sensitive to the image being completely loaded
        }
};