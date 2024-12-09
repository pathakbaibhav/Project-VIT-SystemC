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

/** 
 * WEIGHTS INFO
 * 
 * The weights for the patch embedding layer are saved in four seperate .csv files
 * within the ../tmp directory. The original weights are of the shapes listed out here.
 * In the script to load the weights, multidimensional arrays are flattened to 1D arrays.
 * 
 * cls_token, shape: torch.Size([1, 1, 768]), ndim: 3
 * pos_embed, shape: torch.Size([1, 197, 768]), ndim: 3
 * patch_embed.proj.weight, shape: torch.Size([768, 3, 16, 16]), ndim: 4
 * patch_embed.proj.bias, shape: torch.Size([768]), ndim: 1
 */

SC_MODULE(PatchEmbedding) {
    private:
        Eigen::MatrixXf patch_embedding_weights; // Embedding projection weights
        Eigen::VectorXf cls_token;              // Class token weights
        Eigen::VectorXf pos_embed;              // Positional embedding weights

        std::vector<cv::Mat> extractPatches(const cv::Mat& image);
        Eigen::MatrixXf addPositionEmbeddings(const std::vector<cv::Mat>& patches);
        Eigen::VectorXf patchToVector(const cv::Mat& patch);
        Eigen::VectorXf applyPatchEmbedding(const Eigen::VectorXf& patch);

    public:
    /** Input/Output */
        std::string img_path;           // Path to the input image
        std::string weights_dir;        // Directory with all of the weight files
        float* output_buffer;           // Array to store the output of the patch embedding
        int patch_size;
        int embed_dim;

        sc_in<bool> start;              // Start signal
        sc_in<int> img_length;          // Will always be 256
        sc_in<int> img_height;          // Will always be 256
        sc_in<int> img_channels;        // Will always be 3

        sc_out<bool> done;              // Done signal

        void run();                     // Perform patch embedding

        SC_CTOR(PatchEmbedding) {
            SC_THREAD(run);
        }
};