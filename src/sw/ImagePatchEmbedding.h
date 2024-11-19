#ifndef IMAGE_PATCH_EMBEDDING_H
#define IMAGE_PATCH_EMBEDDING_H

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

// Declare the external patch embedding weights array
extern const float patch_embed_proj_weight[]; // Defined in weights.cpp

extern const float patch_embed_proj_bias []; 

extern const float pos_embed[]; 

extern const float cls_token []; 

#endif // IMAGE_PATCH_EMBEDDING_H
