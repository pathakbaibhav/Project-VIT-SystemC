/**
 * @file patchEmbedding.h
 * @brief Patch embedding layer of the vision transformer
 * @version 0.1
 * @date 2024-12-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

// opencv and systemc have conflicting definitions for int64 and uint64. We use systemc's definitions (long long)
#define int64 opencv_broken_int
#define uint64 opencv_broken_uint
#include <opencv2/opencv.hpp>
#undef int64
#undef uint64

#include <systemc>
#include <vector>
#include <Eigen/Dense>

using namespace sc_core;

SC_MODULE(PatchEmbedding) {
    private:
        // Weights
        Eigen::MatrixXf patch_embed_proj_weight;    // Shape: [768, 3 * 16 * 16]
        Eigen::VectorXf patch_embed_proj_bias;      // Shape: [768]
        Eigen::MatrixXf pos_embed;                  // Shape: [197, 768]
        Eigen::MatrixXf cls_token;                  // Shape: [1, 768]

        cv::Mat getImage(const std::string& image_path);
        Eigen::MatrixXf extractPatches(const cv::Mat& image);
        Eigen::MatrixXf patchEmbeddingMain();

    public:
        std::string img_path;           // Path to the input image
        std::string weights_dir;        // Directory with all of the weight files
        Eigen::MatrixXf* output_buffer; // Matrix to store the output of the patch embedding
        int patch_height;               // How many pixels is a patch – using 16x16
        int embed_dim;                  // Dimension of embedding layer – using 768

        sc_in<bool> start;              // Start signal
        sc_in<int> img_length;          // Will always be 224
        sc_in<int> img_height;          // Will always be 224
        sc_in<int> img_channels;        // Will always be 3

        sc_out<bool> done;              // Done signal

        void run();                     // Perform patch embedding

        SC_CTOR(PatchEmbedding) {
            SC_THREAD(run);
        }
};