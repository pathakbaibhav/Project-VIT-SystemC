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

#include <systemc>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

class ImagePatchEmbedding {
public:
    // Constructor: Initializes the patch size and embedding dimension
    ImagePatchEmbedding(int patchSize, int embedDim);

    // Extract patches from an image
    vector<Mat> extractPatches(const Mat& image);

    // Add position embeddings to the patches
    Eigen::MatrixXf addPositionEmbeddings(const vector<Mat>& patches);

    // Convert an image patch to a vector (flattening the patch)
    Eigen::VectorXf patchToVector(const Mat& patch);

    // Apply patch embedding to a patch (using pre-trained weights)
    Eigen::VectorXf applyPatchEmbedding(const Eigen::VectorXf& patch);

private:
    int patchSize;      // Size of each image patch
    int embedDim;       // Embedding dimension
};



SC_MODULE(PatchEmbedding) {
    public:
    /** Input/Output */
        std::string img_path;           // Path to the input image
        std::string weights_dir;        // Directory with all of the weight files
        int patch_size;
        int embed_dim;

        sc_in<int> img_length;          // Will always be 256
        sc_in<int> img_height;          // Will always be 256
        sc_in<int> img_channels;        // Will always be 3

        sc_out<bool> done;              // Done signal

        void run();                     // Perform patch embedding

        SC_CTOR(PatchEmbedding) {
            SC_METHOD(run);
        }
}