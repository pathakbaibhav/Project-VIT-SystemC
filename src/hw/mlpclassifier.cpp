// src/hw/mlpclassifier.cpp
#include "mlpclassifier.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
SC_MODULE(TopModule)
{
    // Clock and reset signals
    sc_clock clk;
    sc_signal<bool> reset;
    sc_signal<bool> start_pe;
    sc_signal<bool> done_pe;
    sc_signal<bool> start_tr;
    sc_signal<bool> done_tr;
    sc_signal<bool> start_mlp;
    sc_signal<bool> done_mlp;

    // [CLS] Token signals (assuming Transformer outputs [CLS] token)
    sc_signal<sc_fixed<FIXED_POINT_WIDTH, FIXED_POINT_INT_BITS>> cls_token[EMBED_DIM];

    // Classification output signals
    sc_signal<sc_fixed<FIXED_POINT_WIDTH, FIXED_POINT_INT_BITS>> classification[NUM_CLASSES];

    // Module instances
    PatchEmbedding pe;
    Transformer tr;
    MLPClassifier mlp;

    // Constructor
    SC_CTOR(TopModule) : clk("clk", 10, SC_NS),
                         pe("PatchEmbedding"),
                         tr("Transformer"),
                         mlp("MLPClassifier")
    {
        // Connect PatchEmbedding ports
        pe.clk(clk);
        pe.reset(reset);
        pe.start(start_pe);
        pe.done(done_pe);
        // Connect PatchEmbedding output to Transformer input
        // (Assuming appropriate ports and signals)

        // Connect Transformer ports
        tr.clk(clk);
        tr.reset(reset);
        tr.start(start_tr);
        tr.done(done_tr);
        // Connect Transformer output [CLS] token to MLPClassifier input
        for (int i = 0; i < EMBED_DIM; ++i)
        {
            tr.out_cls_token[i](cls_token[i]);
            mlp.cls_token[i](cls_token[i]);
        }

        // Connect MLPClassifier ports
        mlp.clk(clk);
        mlp.reset(reset);
        mlp.start(start_mlp);
        mlp.done(done_mlp);
        for (int i = 0; i < NUM_CLASSES; ++i)
        {
            mlp.classification[i](classification[i]);
        }

        // Define control logic (e.g., start sequencing)
        SC_THREAD(control_logic);
        sensitive << clk.pos();
        dont_initialize();
    }

    // Control logic process
    void control_logic()
    {
        // Initial reset
        reset.write(true);
        wait(20, SC_NS);
        reset.write(false);
        wait(10, SC_NS);

        // Start PatchEmbedding
        start_pe.write(true);
        wait(10, SC_NS);
        start_pe.write(false);

        // Wait for PatchEmbedding to complete
        wait(done_pe.posedge_event());
        wait(); // Wait for the event to propagate

        // Start Transformer
        start_tr.write(true);
        wait(10, SC_NS);
        start_tr.write(false);

        // Wait for Transformer to complete
        wait(done_tr.posedge_event());
        wait(); // Wait for the event to propagate

        // Start MLPClassifier
        start_mlp.write(true);
        wait(10, SC_NS);
        start_mlp.write(false);

        // Wait for MLPClassifier to complete
        wait(done_mlp.posedge_event());
        wait(); // Wait for the event to propagate

        // Optionally, print classification results
        std::cout << "[INFO] Classification Completed. Logits:" << std::endl;
        for (int i = 0; i < NUM_CLASSES; ++i)
        {
            std::cout << "Class " << i << ": " << classification[i].read().to_float() << std::endl;
        }

        // End simulation
        sc_stop();
    }
};

#endif // TOP_MODULE_H
