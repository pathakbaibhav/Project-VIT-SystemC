/**
 * @file transformer.cpp
 * @brief Implementation of transformer block
 * @version 0.1
 * @date 2024-12-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <systemc>
#include <Eigen/Dense>
#include <iostream>

#include "transformer.h"
#include "../include/weights.h"


/****************************************************************************************
 ********************************** TRANSFORMER *****************************************
 ****************************************************************************************/

// Self-Attention Layer
void Transformer::self_attention() {
    try {
        load_weights();

        const Eigen::MatrixXf& input = inpL.read();
        
        // Extract Q, K, V weights and biases
        Eigen::MatrixXf W_q = attn_qkv_weight.block(0, 0, HIDDEN_SIZE, attn_qkv_weight.cols());
        Eigen::MatrixXf W_k = attn_qkv_weight.block(HIDDEN_SIZE, 0, HIDDEN_SIZE, attn_qkv_weight.cols());
        Eigen::MatrixXf W_v = attn_qkv_weight.block(2 * HIDDEN_SIZE, 0, HIDDEN_SIZE, attn_qkv_weight.cols());

        Eigen::VectorXf b_q = attn_qkv_bias.segment(0, HIDDEN_SIZE);
        Eigen::VectorXf b_k = attn_qkv_bias.segment(HIDDEN_SIZE, HIDDEN_SIZE);
        Eigen::VectorXf b_v = attn_qkv_bias.segment(2 * HIDDEN_SIZE, HIDDEN_SIZE);

        // Compute Q, K, V
        Eigen::MatrixXf Q = (W_q * input).colwise() + b_q;
        Eigen::MatrixXf K = (W_k * input).colwise() + b_k;
        Eigen::MatrixXf V = (W_v * input).colwise() + b_v;

        // Compute scaled dot-product attention
        Eigen::MatrixXf attention_weights = Q.transpose() * K / std::sqrt(static_cast<float>(HEAD_DIM));
        attention_weights = attention_weights.array().exp();
        // attention_weights = attention_weights.colwise() / attention_weights.rowwise().sum();
        attention_weights.array().colwise() /= attention_weights.rowwise().sum().array();


        // Compute attention output
        Eigen::MatrixXf attention_output = V * attention_weights.transpose();

        // Apply projection weights and biases
        attention_output = (attn_proj_weight * attention_output).colwise() + attn_proj_bias;

        // Layer normalization
        attention_output = (attention_output.rowwise() - attention_output.colwise().mean());
        attention_output = attention_output.array().rowwise() / attention_output.colwise().squaredNorm().array().sqrt();
        attention_output = (attention_output.array().colwise() * norm1_bias.array());

        cur.write(attention_output);
        // debug_matrix(attention_output, "Attention Output");
        // return attention_output;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Self-Attention failed: " << e.what() << "\n";
    }
}

// Feed-Forward Network
void Transformer::feed_forward() {
    try {
        const Eigen::MatrixXf& input = inpFF.read();

        // First feed-forward layer
        Eigen::MatrixXf ff_hidden = (mlp_fc1_weight * input).colwise() + mlp_fc1_bias;
        ff_hidden = ff_hidden.unaryExpr([](float x) { return std::tanh(x); }); // GELU activation

        // Second feed-forward layer
        Eigen::MatrixXf ff_output = (mlp_fc2_weight * ff_hidden).colwise() + mlp_fc2_bias;

        // Layer normalization
        ff_output = (ff_output.rowwise() - ff_output.colwise().mean());
        ff_output = ff_output.array().rowwise() / ff_output.colwise().squaredNorm().array().sqrt();
        ff_output = (ff_output.array().colwise() * norm2_weight.array()).colwise() + norm2_bias.array();

        cur.write(ff_output);
        // debug_matrix(ff_output, "Feed-Forward Output");
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Feed-Forward failed: " << e.what() << "\n";
    }
}

// Residual Connections and Final Output
void Transformer::compute_output() {
    try {
        Eigen::MatrixXf residual = inpL.read();
        Eigen::MatrixXf output = cur.read() + residual; // Add residual connection
        cur.write(output);

        // debug_matrix(output, "Final Output");
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Compute Output failed: " << e.what() << "\n";
    }
}

void Transformer::load_weights() {
    // Load the weights
    norm1_bias = getWeights(weights_dir + "/trBlock0/norm1_weight.csv", HIDDEN_SIZE);
    attn_qkv_weight = getWeights(weights_dir + "/trBlock0/attn_qkv_weight.csv", 3 * HIDDEN_SIZE, HIDDEN_SIZE);
    attn_qkv_bias = getWeights(weights_dir + "/trBlock0/attn_qkv_bias.csv", 3 * HIDDEN_SIZE);
    attn_proj_weight = getWeights(weights_dir + "/trBlock0/attn_proj_weight.csv", HIDDEN_SIZE, HIDDEN_SIZE);
    attn_proj_bias = getWeights(weights_dir + "/trBlock0/attn_proj_bias.csv", HIDDEN_SIZE);
    norm2_weight = getWeights(weights_dir + "/trBlock0/norm2_weight.csv", HIDDEN_SIZE);
    norm2_bias = getWeights(weights_dir + "/trBlock0/norm2_bias.csv", HIDDEN_SIZE);
    mlp_fc1_weight = getWeights(weights_dir + "/trBlock0/mlp_fc1_weight.csv", 4 * HIDDEN_SIZE, HIDDEN_SIZE);
    mlp_fc1_bias = getWeights(weights_dir + "/trBlock0/mlp_fc1_bias.csv", 4 * HIDDEN_SIZE);
    mlp_fc2_weight = getWeights(weights_dir + "/trBlock0/mlp_fc2_weight.csv", HIDDEN_SIZE, 4 * HIDDEN_SIZE);
    mlp_fc2_bias = getWeights(weights_dir + "/trBlock0/mlp_fc2_bias.csv", HIDDEN_SIZE);
}

// void Transformer::transformerMain() {
//     inpL.write(*input_buffer);
//     load_weights();
//     self_attention();
//     feed_forward();
//     compute_output();
// }

/****************************************************************************************
 *********************************** SYSTEMC ********************************************
 ****************************************************************************************/

void Transformer::run() {
    while(true) {
        // Wait for the start signal
        wait(start.posedge_event());

        std::cout << "Starting transformer" << std::endl;
        // load_weights();

        // transformerMain();
        // inpL.write(*input_buffer);
        // self_attention();
        // feed_forward();
        // compute_output();

        // Send done signal
        done = true;
    }
}