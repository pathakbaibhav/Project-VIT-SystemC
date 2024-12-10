/**
 * @file patchEmbedding.cpp
 * @brief Functions to perform the patch embedding of vision transformer
 * @version 0.1
 * @date 2024-12-06
 * 
 * @copyright Copyright (c) 2024
 */

#include <systemc>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <Eigen/Dense>

#include "patchEmbedding.h"
#include "../include/weights.h"

using namespace std;
using namespace cv;


/****************************************************************************************
 ********************************** PATCH EMBEDDING *************************************
 ****************************************************************************************/

cv::Mat PatchEmbedding::getImage(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);      // Convert from BGR to RGB (PyTorch uses RGB)
    cv::resize(image, image, cv::Size(224, 224));       // Resize image to 224x224
    image.convertTo(image, CV_32F, 1.0 / 255.0);        // Convert to float and normalize

    return image;
}

Eigen::MatrixXf PatchEmbedding::extractPatches(const cv::Mat& image) {
    const int num_patches = (img_height / patch_height)*(img_height / patch_height);    // Number of patches -> 14x14 = 196 total
    const int patch_size = 16 * 16 * 3;                 // 16x16 patch size with 3 channels

    Eigen::MatrixXf patches(num_patches, patch_size);

    // Split into 16x16 patches and flatten each patch
    int patch_idx = 0;
    for (int y = 0; y < image.rows; y += 16) {
        for (int x = 0; x < image.cols; x += 16) {
            // Extract the patch
            cv::Mat patch = image(cv::Rect(x, y, 16, 16));

            // Flatten the patch (HWC -> 1D)
            Eigen::VectorXf flattened_patch(patch_size);
            int idx = 0;
            for (int c = 0; c < 3; ++c) { // Channels
                for (int i = 0; i < 16; ++i) { // Height
                    for (int j = 0; j < 16; ++j) { // Width
                        flattened_patch(idx++) = patch.at<cv::Vec3f>(i, j)[c];
                    }
                }
            }

            // Store the flattened patch in the Eigen matrix
            patches.row(patch_idx++) = flattened_patch;
        }
    }

    return patches;
}

Eigen::MatrixXf PatchEmbedding::patchEmbeddingMain() {
    // Load the weights
    patch_embed_proj_weight = getWeights(weights_dir + "/embedding/patch_embed_proj_weight.csv", 768, 3 * 16 * 16);
    patch_embed_proj_bias = getWeights(weights_dir + "/embedding/patch_embed_proj_bias.csv", 768);
    cls_token = getWeights(weights_dir + "/embedding/cls_token.csv", 1, 768);
    pos_embed = getWeights(weights_dir + "/embedding/pos_embed.csv", 197, 768);

    // DEBUG
    std::cout << "patch_embed_proj_weight dimensions: " << patch_embed_proj_weight.rows() << "x" << patch_embed_proj_weight.cols() << std::endl;
    
    cv::Mat image = getImage(img_path);                 // Load the image
    Eigen::MatrixXf patches = extractPatches(image);    // Extract patches

    // DEBUG
    std::cout << "patches dimensions: " << patches.rows() << "x" << patches.cols() << std::endl;

    // Embedding with weights from the pretrained model
    Eigen::MatrixXf X_embedded = patches * patch_embed_proj_weight.transpose();     // Patch embedding weights
    X_embedded.rowwise() += patch_embed_proj_bias.transpose();                      // Patch embedding bias
    X_embedded.conservativeResize(X_embedded.rows()+1, X_embedded.cols());          // Prepend CLS token
    X_embedded.row(0) = cls_token;
    X_embedded += pos_embed;                                                        // Add positional embedding

    // DEBUG
    std::cout << "X_embedded dimensions: " << X_embedded.rows() << "x" << X_embedded.cols() << std::endl;


    return X_embedded;
}

/****************************************************************************************
 *********************************** SYSTEMC ********************************************
 ****************************************************************************************/

void PatchEmbedding::run() {
    while (true) {
        // Wait for the start signal
        wait(start.posedge_event());

        std::cout << "Starting patch embedding" << std::endl;
        *output_buffer = patchEmbeddingMain();

        // Send done signal
        done = true;
    }
}
