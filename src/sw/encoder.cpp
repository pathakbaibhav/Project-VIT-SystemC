#include <systemc.h> 
#include <iostream> 
#include <cmath> 
#include <Eigen/Dense> 
#include <vector> 
#include <ctime> 
#include <random>
#include "encoder.h"

const int num_heads = 12;
const int embed_size = 768; 
const int n = 196; 
const int head_dim = embed_size / num_heads;

Eigen::MatrixXf tensor(n + 1, embed_size);

void LayerNorm(Eigen::MatrixXf &input) {
    Eigen::VectorXf mean = input.rowwise().mean(); 
    Eigen::MatrixXf centered = input.colwise() - mean; 
    Eigen::VectorXf variance = (centered.array().square().rowwise().mean()); 
    Eigen::VectorXf stddev = (variance.array() + 1e-6).sqrt(); 
    input = centered.array().colwise() / stddev.array();
}


// need to replace by weights
void initialize_weights(Eigen::MatrixXf &Wq, Eigen::MatrixXf &Wk, Eigen::MatrixXf &Wv, Eigen::MatrixXf &Wo) {
    Wq = Eigen::MatrixXf::Random(embed_size, embed_size);
    Wk = Eigen::MatrixXf::Random(embed_size, embed_size);
    Wv = Eigen::MatrixXf::Random(embed_size, embed_size);
    Wo = Eigen::MatrixXf::Random(embed_size, embed_size);
}

void multi_head_attention(Eigen::MatrixXf &tensor) {
    // linear projection to get matrix Q, K, V
    Eigen::MatrixXf Wq(embed_size, embed_size), Wk(embed_size, embed_size), Wv(embed_size, embed_size), Wo(embed_size, embed_size);
    initialize_weights(Wq, Wk, Wv, Wo);

    Eigen::MatrixXf Q = tensor * Wq;
    Eigen::MatrixXf K = tensor * Wk;
    Eigen::MatrixXf V = tensor * Wv;

    // split into different heads
    std::vector<Eigen::MatrixXf> Q_heads(num_heads), K_heads(num_heads), V_heads(num_heads);
    for (int i = 0; i < num_heads; ++i) {
        Q_heads[i] = Q.middleCols(i * head_dim, head_dim);
        K_heads[i] = K.middleCols(i * head_dim, head_dim);
        V_heads[i] = V.middleCols(i * head_dim, head_dim);
    }

    // scaled dot-product attention for each head
    std::vector<Eigen::MatrixXf> attention_heads(num_heads);
    for (int i = 0; i < num_heads; ++i) {
        Eigen::MatrixXf scores = Q_heads[i] * K_heads[i].transpose() / sqrt(static_cast<float>(head_dim));
        Eigen::MatrixXf attention_weights = scores.array().exp();
        //attention_weights = attention_weights.array().rowwise() / attention_weights.rowwise().sum().array();
        attention_weights = attention_weights.array().rowwise() / attention_weights.rowwise().sum().array().transpose();

        attention_heads[i] = attention_weights * V_heads[i];
    }

    // concatenate heads and apply final linear projection
    Eigen::MatrixXf concatenated_attention(n + 1, embed_size);
    for (int i = 0; i < num_heads; ++i) {
        concatenated_attention.middleCols(i * head_dim, head_dim) = attention_heads[i];
    }

    tensor = concatenated_attention * Wo;
}

// void Dropout(Eigen::MatrixXf &input, float dropout_rate) {
//     std::srand(static_cast<unsigned>(std::time(0)));
//     Eigen::MatrixXf mask = Eigen::MatrixXf::Random(input.rows(), input.cols()).unaryExpr([dropout_rate](float x) -> float {
//         return (x > dropout_rate) ? 1.0f : 0.0f;
//     });
//     input = input.array() * mask.array();
// }

Eigen::MatrixXf residual_connection(const Eigen::MatrixXf &input, const Eigen::MatrixXf &output) {
    return input + output;
}

void GELU(Eigen::MatrixXf &matrix) {
    matrix = matrix.unaryExpr([](float x) -> float { 
        return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3)))); 
    });
}

void MLPBlock(Eigen::MatrixXf &input_tensor) {
    // feed forward network, should replace with weights
    Eigen::MatrixXf W1 = Eigen::MatrixXf::Random(embed_size, embed_size * 4); // Expand dimensionality
    Eigen::MatrixXf W2 = Eigen::MatrixXf::Random(embed_size * 4, embed_size); // Reduce dimensionality
    Eigen::VectorXf b1 = Eigen::VectorXf::Random(embed_size * 4);            // Bias for first layer
    Eigen::VectorXf b2 = Eigen::VectorXf::Random(embed_size);                // Bias for second layer

    // first linear layer with GELU
    Eigen::MatrixXf hidden = (input_tensor * W1).rowwise() + b1.transpose();
    GELU(hidden);
    //Dropout(hidden, 0.1f); 

    // second linear layer
    Eigen::MatrixXf output = (hidden * W2).rowwise() + b2.transpose();
    tensor = residual_connection(input_tensor,output);
    LayerNorm(tensor);
}


int sc_main(int argc, char* argv[]) {
    tensor.setRandom();
    Eigen::MatrixXf input_tensor = tensor;
    LayerNorm(tensor);

    multi_head_attention(tensor);

    //Dropout(tensor, 0.1f);     
    tensor = residual_connection(input_tensor, tensor);
    LayerNorm(tensor);

    MLPBlock(tensor);

    return 0;
}



