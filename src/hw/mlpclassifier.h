// src/hw/mlpclassifier.h
#ifndef MLP_CLASSIFIER_H
#define MLP_CLASSIFIER_H

#include <systemc.h>
#include <vector>

// Define constants based on your model
#define EMBED_DIM 768    // D
#define HIDDEN_DIM 3072  // H (e.g., 4 * D)
#define NUM_CLASSES 1000 // K (e.g., ImageNet has 1000 classes)
#define FIXED_POINT_WIDTH 16
#define FIXED_POINT_INT_BITS 8

SC_MODULE(MLPClassifier)
{
    // Ports
    sc_in<bool> clk;                                                                       // Clock signal
    sc_in<bool> reset;                                                                     // Reset signal
    sc_in<bool> start;                                                                     // Start signal
    sc_out<bool> done;                                                                     // Done signal
    sc_in<sc_fixed<FIXED_POINT_WIDTH, FIXED_POINT_INT_BITS>> cls_token[EMBED_DIM];         // [CLS] Token input
    sc_out<sc_fixed<FIXED_POINT_WIDTH, FIXED_POINT_INT_BITS>> classification[NUM_CLASSES]; // Classification output

    // Internal variables for weights and biases
    std::vector<std::vector<float>> W1; // Shape: HIDDEN_DIM x EMBED_DIM
    std::vector<float> b1;              // Shape: HIDDEN_DIM
    std::vector<std::vector<float>> W2; // Shape: NUM_CLASSES x HIDDEN_DIM
    std::vector<float> b2;              // Shape: NUM_CLASSES

    // Methods
    void load_weights();
    void linear_layer1(std::vector<float> & output, const std::vector<float> &input);
    void activation_gelu(std::vector<float> & input_output);
    void linear_layer2(std::vector<float> & output, const std::vector<float> &input);
    void run();

    SC_CTOR(MLPClassifier)
    {
        SC_THREAD(run);
        sensitive << clk.pos();
        dont_initialize();
    }
};

#endif // MLP_CLASSIFIER_H
