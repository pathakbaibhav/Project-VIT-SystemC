### Task 3 (1 person): MLP Classification Block

[Task 3 (1 person): MLP Classification Block](#task-3-1-person-mlp-classification-block)

Uses fully connected architecture to produce classification 

**Input:** (N+1) x D, output of the transformer block
**Input:** Weights+bias from pretrained model
**Output:** Scalar classification
**Operation:** Fully connected layers to produce the classification. Heavily relies on GEMM, so maybe can experiment with our fixed-point implementation here🤷

Impoartant Links: 1) https://viso.ai/deep-learning/vision-transformer-vit/
2) https://medium.com/@hassaanidrees7from-transformers-to-vision-transformers-vit-applying-nlp-models-to-computer-vision-fe6f13b4d014
3)https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6

CNN vs VIT
CNN uses pixel arrays, whereas ViT splits the input images into visual tokens.The visual transformer divides an image into fixed-size patches, correctly embeds each of them, and includes positional embedding as an input to the transformer encoder. Moreover, ViT models outperform CNNs by almost four times when it comes to computational efficiency and accuracy.

**Vision Transformer ViT Architecture**
We can find several proposals for vision transformer models in the literature. The overall structure of the vision transformer architecture consists of the following steps:

Split an image into patches (fixed sizes)
Flatten the image patches
Create lower-dimensional linear embeddings from these flattened image patches
Include positional embeddings
Feed the sequence as an input to a SOTA transformer encoder
Pre-train the ViT model with image labels, then fully supervised on a big dataset
Fine-tune the downstream dataset for image classification