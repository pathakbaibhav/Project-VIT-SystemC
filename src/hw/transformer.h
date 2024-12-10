/**
 * @file transformer.h
 * @brief Transformer block
 * @version 0.1
 * @date 2024-12-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <systemc>
#include <Eigen/Dense>

const int HIDDEN_SIZE = 768;
const int NUM_HEADS = 12;
const int HEAD_DIM = HIDDEN_SIZE / NUM_HEADS;

using namespace sc_core;

SC_MODULE(Transformer) {
    private:
        // Weights
        Eigen::VectorXf norm1_bias;                 // Shape: [768]
        Eigen::MatrixXf attn_qkv_weight;            // Shape: [2304, 768]
        Eigen::VectorXf attn_qkv_bias;              // Shape: [2304]
        Eigen::MatrixXf attn_proj_weight;           // Shape: [768, 768]
        Eigen::VectorXf attn_proj_bias;             // Shape: [768]
        Eigen::VectorXf norm2_weight;               // Shape: [768]
        Eigen::VectorXf norm2_bias;                 // Shape: [768]
        Eigen::MatrixXf mlp_fc1_weight;             // Shape: [3072, 768]
        Eigen::VectorXf mlp_fc1_bias;               // Shape: [3072]
        Eigen::MatrixXf mlp_fc2_weight;             // Shape: [768, 3072]
        Eigen::VectorXf mlp_fc2_bias;               // Shape: [768]
        
        void self_attention();                      // Self-Attention Layer
        void feed_forward();                        // Feed-Forward Network
        void compute_output();                      // Compute the output
        void load_weights();                        // Load the weights
        void transformerMain();                     // Main function for the transformer block   

    public:
        std::string weights_dir;        // Directory with all of the weight files
        Eigen::MatrixXf* input_buffer;  // Holds the input
        Eigen::MatrixXf* output_buffer; // Holds the output
        int embed_dim;                  // Dimension of embedding layer – using 768

        sc_in<bool> start;              // Start signal
        sc_out<bool> done;              // Done signal

        sc_signal<Eigen::MatrixXf> inpL, inpFF;
        sc_signal<Eigen::MatrixXf> cur;

        void run();

        SC_CTOR(Transformer) {
            SC_THREAD(run);

            SC_METHOD(self_attention);
            sensitive << inpL;

            SC_METHOD(feed_forward);
            sensitive << inpFF;

            SC_METHOD(compute_output);
            sensitive << cur;
        }
};