#ifndef ENCODER_H
#define ENCODER_H

#include <systemc.h>
#include <Eigen/Dense>

// Define dimensions as constants for simplicity
const int HIDDEN_SIZE = 768;
const int NUM_HEADS = 12;
const int HEAD_DIM = HIDDEN_SIZE / NUM_HEADS;

SC_MODULE(encoder) {
    // Input and output signals
    sc_signal<Eigen::MatrixXf> inpL, inpFF;
    sc_signal<Eigen::MatrixXf> cur;

    // Encoder operations
    void self_attention();
    void feed_forward();
    void compute_output();

    // Constructor
    SC_CTOR(encoder) {
        SC_METHOD(self_attention);
        sensitive << inpL;

        SC_METHOD(feed_forward);
        sensitive << inpFF;

        SC_METHOD(compute_output);
        sensitive << cur;
    }
};

#endif // ENCODER_H