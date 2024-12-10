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
Eigen::MatrixXf Transformer::self_attention() {
    load_weights();

    // const Eigen::MatrixXf& input = inpL.read();
    const Eigen::MatrixXf& input = *input_buffer;
    
    // Extract Q, K, V weights and biases
    Eigen::MatrixXf W_q = attn_qkv_weight.block(0, 0, HIDDEN_SIZE, attn_qkv_weight.cols());
    Eigen::MatrixXf W_k = attn_qkv_weight.block(HIDDEN_SIZE, 0, HIDDEN_SIZE, attn_qkv_weight.cols());
    Eigen::MatrixXf W_v = attn_qkv_weight.block(2 * HIDDEN_SIZE, 0, HIDDEN_SIZE, attn_qkv_weight.cols());

    Eigen::VectorXf b_q = attn_qkv_bias.segment(0, HIDDEN_SIZE);
    Eigen::VectorXf b_k = attn_qkv_bias.segment(HIDDEN_SIZE, HIDDEN_SIZE);
    Eigen::VectorXf b_v = attn_qkv_bias.segment(2 * HIDDEN_SIZE, HIDDEN_SIZE);

    // Compute Q, K, V
    Eigen::MatrixXf Q = (input * W_q.transpose()).rowwise() + b_q.transpose();
    Eigen::MatrixXf K = (input * W_k.transpose()).rowwise() + b_k.transpose();
    Eigen::MatrixXf V = (input * W_v.transpose()).rowwise() + b_v.transpose();

    // Compute scaled dot-product attention
    Eigen::MatrixXf attention_weights = Q.transpose() * K / std::sqrt(static_cast<float>(HEAD_DIM));
    attention_weights = attention_weights.array().exp();
    // attention_weights = attention_weights.colwise() / attention_weights.rowwise().sum();
    attention_weights.array().colwise() /= attention_weights.rowwise().sum().array();

    // Compute attention output
    Eigen::MatrixXf attention_output = V * attention_weights.transpose();

    // Apply projection weights and biases
    attention_output = (attention_output * attn_proj_weight.transpose()).rowwise() + attn_proj_bias.transpose();

    // Layer normalization
    attention_output = (attention_output.rowwise() - attention_output.colwise().mean());
    attention_output = attention_output.array().rowwise() / attention_output.colwise().squaredNorm().array().sqrt();
    attention_output = (attention_output.array().rowwise() * norm1_bias.transpose().array());

    // cur.write(attention_output);
    return attention_output;
}

// Feed-Forward Network
Eigen::MatrixXf Transformer::feed_forward(const Eigen::MatrixXf input) {
    // const Eigen::MatrixXf& input = inpFF.read();

    // First feed-forward layer
    Eigen::MatrixXf ff_hidden = (input * mlp_fc1_weight.transpose()).rowwise() + mlp_fc1_bias.transpose();
    ff_hidden = ff_hidden.unaryExpr([](float x) { return std::tanh(x); }); // GELU activation

    // Second feed-forward layer
    Eigen::MatrixXf ff_output = (ff_hidden * mlp_fc2_weight.transpose()).rowwise() + mlp_fc2_bias.transpose();

    // Layer normalization
    ff_output = (ff_output.rowwise() - ff_output.colwise().mean());
    ff_output = ff_output.array().rowwise() / ff_output.colwise().squaredNorm().array().sqrt();
    ff_output = (ff_output.array().rowwise() * norm2_weight.array().transpose()).rowwise() + norm2_bias.transpose().array();


    // cur.write(ff_output);
    return ff_output;
}

// Residual Connections and Final Output
Eigen::MatrixXf Transformer::compute_output(Eigen::MatrixXf cur) {
    Eigen::MatrixXf residual = *input_buffer;
    Eigen::MatrixXf output = cur + residual; // Add residual connection
    
    return output;
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

Eigen::MatrixXf Transformer::transformerMain() {
    load_weights();
    return compute_output(feed_forward(self_attention()));
}

/****************************************************************************************
 *********************************** SYSTEMC ********************************************
 ****************************************************************************************/

void Transformer::run() {
    while(true) {
        // Wait for the start signal
        wait(start.posedge_event());

        std::cout << "Starting transformer" << std::endl;
        *output_buffer=transformerMain();

        std::cout << "Output Buffer:\n" << *output_buffer << std::endl;


        // Send done signal
        done = true;
    }
}