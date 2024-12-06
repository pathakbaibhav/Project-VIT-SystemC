#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

// make sure to call the above .so files in makefile 

using namespace std;
using namespace cv;

extern const float patch_embedding_weights[];  // Declared in weights.cpp

// Helper function to extract patches from the image
vector<Mat> extractPatches(const Mat& image, int patchSize) {
    vector<Mat> patches;
    int rows = image.rows / patchSize;
    int cols = image.cols / patchSize;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Extract patch
            Rect roi(j * patchSize, i * patchSize, patchSize, patchSize);
            Mat patch = image(roi);
            patches.push_back(patch);
        }
    }

    return patches;
}


// Function to add positional embeddings to the patches
Eigen::MatrixXf addPositionEmbeddings(const vector<Mat>& patches, int embedDim) {
    int numPatches = patches.size();
    Eigen::MatrixXf embeddings(numPatches, embedDim);

    // Initialize position embeddings as sinusoidal functions
    for (int i = 0; i < numPatches; ++i) {
        // Get 2D position of the patch
        int row = i / (patches.size() / sqrt(numPatches));
        int col = i % (patches.size() / sqrt(numPatches));

        for (int j = 0; j < embedDim; ++j) {
            float angle = (float)(i) / pow(10000, 2 * (j / 2) / (float)embedDim);
            if (j % 2 == 0) {
                embeddings(i, j) = sin(angle);
            } else {
                embeddings(i, j) = cos(angle);
            }
        }
    }

    return embeddings;
}

// Function to convert patch to a vector (flatten it)
Eigen::VectorXf patchToVector(const Mat& patch) {
    // Flatten the patch (convert it to a vector)
    Eigen::VectorXf patchVec(patch.rows * patch.cols * patch.channels());
    for (int i = 0; i < patch.rows; ++i) {
        for (int j = 0; j < patch.cols; ++j) {
            for (int k = 0; k < patch.channels(); ++k) {
                patchVec(i * patch.cols * patch.channels() + j * patch.channels() + k) = patch.at<Vec3b>(i, j)[k];
            }
        }
    }
    return patchVec;
}



// Function to apply patch embedding to a patch (using the weights)
Eigen::VectorXf applyPatchEmbedding(const Eigen::VectorXf& patch) {
    Eigen::VectorXf embedding(embedDim);

    // Assuming patch_embedding_weights is a 2D array (embed_dim x patch_size^2 * channels)
    for (int i = 0; i < embedDim; ++i) {
        embedding(i) = 0;
        for (int j = 0; j < patch.size(); ++j) {
            embedding(i) += patch(j) * patch_embedding_weights[i * patch.size() + j];
        }
    }

    return embedding;
}


// Main function
int main() {
    // Load the image
    Mat image = imread("image.jpg");
    if (image.empty()) {
        cerr << "Image not found!" << endl;
        return -1;
    }

    // Parameters
    int patchSize = 16;  // Define patch size (e.g., 16x16)
    int embedDim = 768;  // Embedding dimension

    // 1. Extract patches
    vector<Mat> patches = extractPatches(image, patchSize);

    // 2. Convert patches to vectors 
    // - Flattening the 16x16 to 256x1 ? 
    vector<Eigen::VectorXf> patchVectors;
    for (const auto& patch : patches) {
        Eigen::VectorXf patchVec = patchToVector(patch);
        patchVectors.push_back(patchVec);
    }

    // 3. Add position embeddings
    Eigen::MatrixXf positionEmbeddings = addPositionEmbeddings(patches, embedDim);

    // 4. Combine patch embeddings and position embeddings
    int numPatches = patches.size();
    Eigen::MatrixXf combinedEmbeddings(numPatches, embedDim);

    for (int i = 0; i < numPatches; ++i) {
        combinedEmbeddings.row(i) = patchVectors[i].transpose() + positionEmbeddings.row(i);
    }

    // Class embedding ? 
    

    // Output the shape of combined embeddings (for debugging) Is it adhering to the N
    cout << "Combined Embeddings Shape: " << combinedEmbeddings.rows() << " x " << combinedEmbeddings.cols() << endl;

    return 0;
}
