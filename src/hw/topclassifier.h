/**
 * @file testbench.cpp
 * @brief
 * @version 0.1
 * @date 2024-12-03
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef TOPCLASSIFIER_H
#define TOPCLASSIFIER_H

#define SC_INCLUDE_FX
#include <systemc.h>
#include <sysc/datatypes/fx/sc_fixed.h>
#include "patchEmbedding.h"
#include "transformer.h"
#include "mlpclassifier.h"

SC_MODULE(TopModule)
{
    // Clock and reset signals
    sc_in_clk clk;
    sc_signal<bool> reset;
    sc_signal<bool> start_pe, done_pe, start_tr, done_tr, start_mlp, done_mlp;

    // [CLS] Token signals
    sc_signal<sc_fixed<FIXED_POINT_WIDTH, FIXED_POINT_INT_BITS>> cls_token[EMBED_DIM];

    // Classification output signals
    sc_signal<sc_fixed<FIXED_POINT_WIDTH, FIXED_POINT_INT_BITS>> classification[NUM_CLASSES];

    // Module instances
    PatchEmbedding pe;
    Transformer tr;
    MLPClassifier mlp;

    // Constructor declaration
    SC_CTOR(TopModule);

    // Method declarations
    void control_logic();
};

#endif // TOPCLASSIFIER_H