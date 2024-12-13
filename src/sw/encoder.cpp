#include <systemc.h>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "encoder.h"


SC_MODULE(Encoder) {
    sc_in<bool> run_encoder_signal;
    sc_in<Eigen::MatrixXf> input_tensor;
    sc_out<Eigen::MatrixXf> output_tensor;

    //define the weights
    Eigen::MatrixXf Q, K, V;
    Eigen::MatrixXf W1, W2;
    Eigen::VectorXf B1, B2;
    Eigen::MatrixXf tensor;

    //accept the weights when Robbie call this function
    void initialize_weights(const EncoderWeights& weights) {
        Q = weights.Q;
        K = weights.K;
        V = weights.V;
        W1 = weights.mlp_fc1_weight;
        B1 = weights.mlp_fc1_bias;
        W2 = weights.mlp_fc2_weight;
        B2 = weights.mlp_fc2_bias;
    }

    void layer_norm(Eigen::MatrixXf& input) {
        Eigen::VectorXf mean = input.rowwise().mean();
        Eigen::MatrixXf centered = input.colwise() - mean;
        Eigen::VectorXf variance = (centered.array().square().rowwise().mean());
        Eigen::VectorXf stddev = (variance.array() + 1e-6).sqrt();
        input = centered.array().colwise() / stddev.array();
    }

    void multi_head_attention(Eigen::MatrixXf& tensor) {
        const int num_heads = 12;
        const int head_dim = Q.cols() / num_heads;
        std::vector<Eigen::MatrixXf> Q_heads(num_heads), K_heads(num_heads), V_heads(num_heads);
        for (int i = 0; i < num_heads; ++i) {
            Q_heads[i] = Q.middleCols(i * head_dim, head_dim);
            K_heads[i] = K.middleCols(i * head_dim, head_dim);
            V_heads[i] = V.middleCols(i * head_dim, head_dim);
        }

        std::vector<Eigen::MatrixXf> attention_heads(num_heads);
        for (int i = 0; i < num_heads; ++i) {
            Eigen::MatrixXf scores = Q_heads[i] * K_heads[i].transpose() / sqrt(static_cast<float>(head_dim));
            Eigen::MatrixXf attention_weights = scores.array().exp();
            attention_weights = attention_weights.array().rowwise() / attention_weights.rowwise().sum().array();
            attention_heads[i] = attention_weights * V_heads[i];
        }

        Eigen::MatrixXf concatenated_attention(tensor.rows(), tensor.cols());
        for (int i = 0; i < num_heads; ++i) {
            concatenated_attention.middleCols(i * head_dim, head_dim) = attention_heads[i];
        }

        tensor = concatenated_attention;
    }

    Eigen::MatrixXf residual_connection(const Eigen::MatrixXf& input, const Eigen::MatrixXf& output) {
        return input + output;
    }

    //activation
    void gelu(Eigen::MatrixXf& matrix) {
        matrix = matrix.unaryExpr([](float x) -> float {
            return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
        });
    }

    void mlp_block(Eigen::MatrixXf& tensor) {
        //feed forward layer
        Eigen::MatrixXf hidden = (tensor * W1).rowwise() + B1.transpose();
        gelu(hidden);
        Eigen::MatrixXf output = (hidden * W2).rowwise() + B2.transpose();
        tensor = residual_connection(tensor, output);
        layer_norm(tensor);
    }

    // running encoder
    void run_encoder_process() {
        while (true) {
            wait(run_encoder_signal.posedge()); 

            tensor = input_tensor.read();
            layer_norm(tensor);
            multi_head_attention(tensor);
            tensor = residual_connection(input_tensor, tensor);
            layer_norm(tensor);
            mlp_block(tensor);

            output_tensor.write(tensor);
        }
    }

    SC_CTOR(Encoder) {
        SC_THREAD(run_encoder_process);
        sensitive << run_encoder_signal;
    }

};

