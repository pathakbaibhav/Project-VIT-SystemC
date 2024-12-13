#ifndef ENCODER_H
#define ENCODER_H

#include <systemc.h>
#include <Eigen/Dense>

SC_MODULE(Encoder) {
    sc_in<bool> run_encoder_signal;
    sc_in<Eigen::MatrixXf> input_tensor;
    sc_out<Eigen::MatrixXf> output_tensor;

    // Weights for the encoder
    Eigen::MatrixXf Q, K, V;
    Eigen::MatrixXf W1, W2;
    Eigen::VectorXf B1, B2;
    Eigen::MatrixXf tensor;

    void initialize_weights(const EncoderWeights& weights);

    void layer_norm(Eigen::MatrixXf& input);

    void multi_head_attention(Eigen::MatrixXf& tensor);

    Eigen::MatrixXf residual_connection(const Eigen::MatrixXf& input, const Eigen::MatrixXf& output);

    void gelu(Eigen::MatrixXf& matrix);

    void mlp_block(Eigen::MatrixXf& tensor);

    void run_encoder_process();

    SC_CTOR(Encoder) {
        SC_THREAD(run_encoder_process);
        sensitive << run_encoder_signal;
    }
};

#endif // ENCODER_H
