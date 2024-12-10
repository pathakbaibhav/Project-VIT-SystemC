#ifndef ENCODER_H
#define ENCODER_H

#include <systemc.h>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <ctime>
#include <random>


extern const int num_heads;
extern const int embed_size;
extern const int n;
extern const int head_dim;


extern Eigen::MatrixXf tensor;

void LayerNorm();
void initialize_weights(Eigen::MatrixXf &Wq, Eigen::MatrixXf &Wk, Eigen::MatrixXf &Wv, Eigen::MatrixXf &Wo);
void multi_head_attention(Eigen::MatrixXf &tensor);
void Dropout(Eigen::MatrixXf &input, float dropout_rate);
Eigen::MatrixXf residual_connection(const Eigen::MatrixXf &input, const Eigen::MatrixXf &output);
void GELU(Eigen::MatrixXf &matrix);
void MLPBlock(Eigen::MatrixXf &input_tensor);

#endif 
