### Task 3 (1 person): MLP Classification Block

[Task 3 (1 person): MLP Classification Block](#task-3-1-person-mlp-classification-block)

Uses fully connected architecture to produce classification 

**Input:** (N+1) x D, output of the transformer block
**Input:** Weights+bias from pretrained model
**Output:** Scalar classification
**Operation:** Fully connected layers to produce the classification. Heavily relies on GEMM, so maybe can experiment with our fixed-point implementation here🤷

Impoartant Links: https://viso.ai/deep-learning/vision-transformer-vit/
