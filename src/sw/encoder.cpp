#include "encoder.h"
#include <iostream>
#include <cmath>

// Debugging helper: Print matrix dimensions
void debug_matrix(const Eigen::MatrixXf& mat, const std::string& name) {
    std::cout << "[DEBUG] " << name << " | Shape: (" << mat.rows() << ", " << mat.cols() << ")\n";
}

// Self-Attention Layer
void encoder::self_attention() {
    try {
        const Eigen::MatrixXf& input = inpL.read();
        const int seq_len = input.cols();

        // Initialize Q, K, V weight matrices (random for example)
        Eigen::MatrixXf W_q = Eigen::MatrixXf::Random(HIDDEN_SIZE, HIDDEN_SIZE);
        Eigen::MatrixXf W_k = Eigen::MatrixXf::Random(HIDDEN_SIZE, HIDDEN_SIZE);
        Eigen::MatrixXf W_v = Eigen::MatrixXf::Random(HIDDEN_SIZE, HIDDEN_SIZE);

        // Compute Q, K, V
        Eigen::MatrixXf Q = W_q * input;
        Eigen::MatrixXf K = W_k * input;
        Eigen::MatrixXf V = W_v * input;

        debug_matrix(Q, "Q Matrix");
        debug_matrix(K, "K Matrix");
        debug_matrix(V, "V Matrix");

        // Compute scaled dot-product attention
        Eigen::MatrixXf attention_weights = Q.transpose() * K / std::sqrt(static_cast<float>(HEAD_DIM));
        attention_weights = attention_weights.array().exp();
        attention_weights = attention_weights.colwise() / attention_weights.rowwise().sum();

        // Compute attention output
        Eigen::MatrixXf attention_output = V * attention_weights.transpose();
        cur.write(attention_output);

        debug_matrix(attention_output, "Attention Output");
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Self-Attention failed: " << e.what() << "\n";
    }
}

// Feed-Forward Network
void encoder::feed_forward() {
    try {
        const Eigen::MatrixXf& input = inpFF.read();

        // Initialize feed-forward weights and biases
        Eigen::MatrixXf W1 = Eigen::MatrixXf::Random(HIDDEN_SIZE, 4 * HIDDEN_SIZE);
        Eigen::MatrixXf b1 = Eigen::MatrixXf::Random(4 * HIDDEN_SIZE, 1);
        Eigen::MatrixXf W2 = Eigen::MatrixXf::Random(4 * HIDDEN_SIZE, HIDDEN_SIZE);
        Eigen::MatrixXf b2 = Eigen::MatrixXf::Random(HIDDEN_SIZE, 1);

        // First layer with GELU activation
        Eigen::MatrixXf ff_hidden = (W1 * input).colwise() + b1.col(0);
        ff_hidden = ff_hidden.unaryExpr([](float x) { return x > 0 ? x : 0; }); // ReLU

        // Second layer
        Eigen::MatrixXf ff_output = (W2 * ff_hidden).colwise() + b2.col(0);
        cur.write(ff_output);

        debug_matrix(ff_output, "Feed-Forward Output");
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Feed-Forward failed: " << e.what() << "\n";
    }
}

// Residual Connections and Final Output
void encoder::compute_output() {
    try {
        Eigen::MatrixXf residual = inpL.read();
        Eigen::MatrixXf output = cur.read() + residual; // Add residual connection
        cur.write(output);

        debug_matrix(output, "Final Output");
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Compute Output failed: " << e.what() << "\n";
    }
}

// Main Simulation
int sc_main(int argc, char* argv[]) {
    encoder e("encoder");

    // Example input tensor (randomized for demonstration)
    Eigen::MatrixXf input = Eigen::MatrixXf::Random(HIDDEN_SIZE, 50); // Sequence length = 50
    e.inpL.write(input);

    // Call methods manually for demonstration
    e.self_attention();
    e.feed_forward();
    e.compute_output();

    // Final output
    Eigen::MatrixXf output = e.cur.read();
    debug_matrix(output, "Simulation Final Output");

    return 0;
}