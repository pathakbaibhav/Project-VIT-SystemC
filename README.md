# PROJECT-VIT_Team

EECE 7368 Project: Modeling a Vision Transformer SoC with SystemC

## Description

In this project, we explore the idea of implementing the Vision Transformer (ViT) image classification tool in pure hardware. We will first implement the exact architecture outlined in Dosovitskiy et. Al. in SystemC. From there, we intend to evaluate our implementation, expand it to work with different pretrained models, and, as a stretch, explore means to produce a synthesizable design of the hardware-ViT that can be deployed on a FPGA.

<details>
<summary>Table of Contents</summary>

- [prj-vit-team](#prj-vit-team)
  - [Description](#description)
  - [Technical Overview](#technical-overview)
    - [Vision Transformer Overview](#vision-transformer-overview)
    - [ViT in System C](#vit-in-system-c)
    - [(Stretch Synthesizable ViT)](#stretch-synthesizable-vit)
  - [Project Breakdown + Distribution of Work](#project-breakdown--distribution-of-work)
    - [Task 1 (1 person): Patch + Position Embedding](#task-1-1-person-patch--position-embedding)
      - [Operation:](#operation)
    - [Task 2 (2 people): Transformer Implementation](#task-2-2-people-transformer-implementation)
    - [Task 3 (1 person): MLP Classification Block](#task-3-1-person-mlp-classification-block)
    - [Task 4 (1 person): Tying everything together](#task-4-1-person-tying-everything-together)
      - [Task 5 (1 person): Expanding the system and developing tests](#task-5-1-person-expanding-the-system-and-developing-tests)
  - [Resources](#resources)
</details>

## Technical Overview

### Vision Transformer Overview

The Vision Transformer (ViT) brings the transformer architecture, popularized by its success in language models, to image classification. Unlike traditional image classification models, ViT does not rely on convolutional layers; instead, it uses a patch embedding and the attention mechanism. An image is divided into non-overlapping patches that are flattened and projected into vectors, effectively treating each patch as a "token." These patch embeddings, combined with positional embeddings to maintain spatial context, are then passed through transformer blocks for global attention across patches.

This patch-based approach makes attention feasible by reducing the number of interactions, which would otherwise scale quadratically with image size if each pixel interacted with all others. To put this into perspective, a 256x256 image would require (256x256)^2 = 4,294,967,296 individual calculations. If we scale this to modern 4K images, we can see that this quickly becomes infeasible for most systems.

We will be implementing the original Vision Transformer architecture from:
  - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

<p align="center">
  <img src="assets/image.png" alt="Vision Transformer overview" width="60%" height="auto">
</p>
<p align="center">
  ViT architecture. Taken from the <a href="https://arxiv.org/abs/2010.11929">original paper</a>.
</p>

### ViT in System C

Why implement this in hardware? For speed—and because it’s an exciting challenge. Our goal is to explore the implementation of the Vision Transformer (ViT) architecture in hardware, beginning with SystemC. Ultimately, we aim to develop a model of a system-on-chip (SoC) that takes an image as input and produces a classification as output. This design will be inference-only, with an additional input for users to load their trained model.

### (Stretch Synthesizable ViT)

Depending on timing, we are interested in trying to create a synthesizable implementation that can be pushed to a FPGA. 

## Project Breakdown + Distribution of Work

### Task 1 (1 person): Patch + Position Embedding - Tilak

Nov19 Notes: 
* Basic skeleton code added and need to configure the patchSize and embedDim based on the pre trained model architecture (e.g., ViT-B/ViT-L). 
* Need to consider having a weights.h instead of weights.cpp ? 


Calculates the embedding for the input image. 

- **Input**: An image of dimension HxWxC = 256x256x3
- **Output**: Sequence of flattened 2D patches of dimension NxP^2xC, where N=number of patches, P=height of each patch (patches are square), C=number of channels (3)
#### Operation: 
★ Resources: 
* See section 3.1 of paper, or 11:00 mark of [this](https://www.youtube.com/watch?v=TrdevFK_am4) video
* See [Springer](https://www.sciencedirect.com/science/article/abs/pii/S1566253524000265) ViT Image patches 

1. Read input image:
    The input is a 256×256×C image, where C is the number of color channels (e.g., C=3 for RGB images).
2. Split Image into Patches:
    Split the 256×256×C image into non-overlapping 16×16×C patches.
    This yields N = (256×256)/(16×16) = 256 patches, each of size 16x16xC.
3. For each 16x16xC patch (Note - this is easily parallelizable!):
    1. Flatten into a vector A of length 16×16×C = 256×C. Each A will be a vector of shape (256×C, 1).
    2. Multiply A by the embedding matrix E to obtain the fixed-length patch embedding B of shape (D,1) for each patch:
        - E is a learnable patch embedding matrix that will be loaded in from a pretrained model. 
        - E is of shape (256×C)×D, where D is the embedding dimension we are using. D is defined by the model.
    3. Add positional embedding P to the patch embedding B:
        - P is a learnable positional embedding matrix that will be loaded in from a pretrained model.
        - P is of shape (D, 1).
        - P comes from a matrix PP, which is of shape NxD, where N is the number of patches and D is the embedding dimension. So, each patch has its own positional embedding vector P. 
    4. Prepend the class token C to the sequence of patch embeddings:
        - C is a learnable class token that will be loaded in from a pretrained model.
        - C is of shape (D, 1).
        - In the ViT architecture visualization, this is represented by the asterisk (*) at position 0 that enters the transformer block
4. Final output:
    - Combine the class embedding and all patch embeddings into a single array. 
    - This output is (N+1) x D, N is number of patches, D is embedding dimension, and the +1 is for the class embedding

### Task 2 (2 people): Transformer Implementation - Luyue

Implementation of the common transformer block

**Input:** (N+1) x D embedding of the original input image
**Input:** Weights, biases, and number of heads, initialized from pretrained model
**Output:** (N+1) x D matrix
**Operation:** Performs multihead attention on the input. This is standard operation now, so just look it up😎

### Task 3 (1 person): MLP Classification Block - Aniket

Uses fully connected architecture to produce classification 

**Input:** (N+1) x D, output of the transformer block
**Input:** Weights+bias from pretrained model
**Output:** Scalar classification
**Operation:** Fully connected layers to produce the classification. Heavily relies on GEMM, so maybe can experiment with our fixed-point implementation here🤷
	
### Task 4 (1 person): Tying everything together - Robbie 
This person is responsible for developing a nice interface to handle the input images and for establishing a convenient and standardized way to load the weights from a pretrained model into each layer. 
They are also responsible for making sure the products from the above tasks function properly with each other. 

### Task 5 (1 person): Expanding the system and developing tests - Baibhav
This person is tasked with making our SystemC implementation more generalized. We will probably center our development around a single pretrained model, so this person will be tasked with exploring ways to make our implementation work with any pretrained model so long as it follows a standardized format.
This person is also tasked with designing and implementing tests that will be used to demonstrate our project’s functionality and performance. 


## Resources
- Dosovitskiy, Alexey. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
  - https://arxiv.org/abs/2010.11929
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Paper Explained)
  - https://www.youtube.com/watch?v=TrdevFK_am4
- vit.cpp
  - https://github.com/staghado/vit.cpp
