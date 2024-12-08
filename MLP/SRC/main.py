# main.py

import torch
from vit import VisionTransformer, preprocess_image
import torch.nn as nn
import sys
import os
import warnings
import torch.nn.functional as F
import math  # <-- Added import

# Suppress the FutureWarning related to torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

# Import utility functions
from utils import rename_state_dict_keys  # Ensure utils.py is in the same directory

def resize_positional_embeddings(checkpoint_embedding, num_patches_model, embed_dim):
    """
    Resize positional embeddings to match the model's expected shape.

    Args:
        checkpoint_embedding (torch.Tensor): Positional embeddings from the checkpoint.
        num_patches_model (int): Number of patches in the model.
        embed_dim (int): Embedding dimension.

    Returns:
        torch.Tensor: Resized positional embeddings.
    """
    num_patches_checkpoint = checkpoint_embedding.size(1) - 1
    cls_token = checkpoint_embedding[:, :1, :]
    patch_embeddings = checkpoint_embedding[:, 1:, :]

    if num_patches_checkpoint == num_patches_model:
        return checkpoint_embedding

    h_checkpoint = int(math.sqrt(num_patches_checkpoint))
    h_model = int(math.sqrt(num_patches_model))

    if h_checkpoint ** 2 != num_patches_checkpoint or h_model ** 2 != num_patches_model:
        raise ValueError("Number of patches must be a perfect square.")

    # Reshape for interpolation
    patch_embeddings = patch_embeddings.view(1, h_checkpoint, h_checkpoint, embed_dim)
    patch_embeddings = patch_embeddings.permute(0, 3, 1, 2)  # (1, D, H_checkpoint, W_checkpoint)

    # Interpolate to new size
    resized_patch_embeddings = F.interpolate(
        patch_embeddings, size=(h_model, h_model), mode='bilinear', align_corners=False
    )

    # Reshape back
    resized_patch_embeddings = resized_patch_embeddings.permute(0, 2, 3, 1).view(1, num_patches_model, embed_dim)

    # Combine [CLS] token with resized patch embeddings
    resized_embeddings = torch.cat((cls_token, resized_patch_embeddings), dim=1)  # (1, N+1, D)

    return resized_embeddings

def get_predicted_class(logits):
    """
    Determines the predicted class from logits.

    Args:
        logits (torch.Tensor): Logits tensor of shape (B, num_classes)
    
    Returns:
        int: Predicted class index
    """
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1)
    return predicted_class.item()

def main():
    # Configuration based on your pre-trained model
    img_size = 256
    patch_size = 16
    embed_dim = 768
    num_heads = 12
    mlp_dim = 3072
    num_layers = 12
    num_classes = 1000  # Adjust based on your dataset

    # Initialize the Vision Transformer
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.1
    )

    # Load custom weights
    state_dict_path = "/home/aniket/EECE7368/prj-vit-team/MLP/SRC/weights/trained_vit.pth"
    if not os.path.exists(state_dict_path):
        print(f"State dict file not found at {state_dict_path}")
        sys.exit(1)

    state_dict = torch.load(state_dict_path, map_location='cpu')  # Removed weights_only=True if not needed
    
    # Handle nested state_dict if present
    if 'state_dict' in state_dict:
        print("Nested 'state_dict' found. Extracting...")
        state_dict = state_dict['state_dict']
    elif 'model_state_dict' in state_dict:
        print("Nested 'model_state_dict' found. Extracting...")
        state_dict = state_dict['model_state_dict']
    else:
        print("No nested state_dict found. Using the loaded state_dict directly.")

    new_state_dict = rename_state_dict_keys(state_dict)
    # Adjust positional embeddings during resizing
    if 'pos_embed.pos_embed' in new_state_dict:
        print("Resizing positional embeddings...")
        checkpoint_embedding = new_state_dict['pos_embed.pos_embed']
        num_patches_model = (img_size // patch_size) ** 2
        resized_pos_embed = resize_positional_embeddings(checkpoint_embedding, num_patches_model, embed_dim)
        new_state_dict['pos_embed.pos_embed'] = resized_pos_embed
        print(f"Resized positional embeddings to shape {resized_pos_embed.size()}")

    # Debug: Print new_state_dict keys
    print("New State Dict Keys:")
    for key in new_state_dict.keys():
        print(key)
    print(f"Total keys in new_state_dict: {len(new_state_dict)}")

    if new_state_dict is None or len(new_state_dict) == 0:
        print("Failed to rename state dict keys or new_state_dict is empty.")
        sys.exit(1)

    load_result = model.load_state_dict(new_state_dict, strict=False)

    # Optional: Initialize missing keys if any
    if load_result.missing_keys:
        print("Missing keys:", load_result.missing_keys)
        for key in load_result.missing_keys:
            if key == 'norm.weight':
                nn.init.ones_(model.norm.weight)
                print(f"Initialized {key} with ones.")
            elif key == 'norm.bias':
                nn.init.zeros_(model.norm.bias)
                print(f"Initialized {key} with zeros.")
            elif key == 'patch_embed.proj.weight':
                # Initialize Conv2d weights if missing
                nn.init.kaiming_normal_(model.patch_embed.proj.weight, mode='fan_out', nonlinearity='relu')
                print(f"Initialized {key} with Kaiming normal.")
            elif key == 'patch_embed.proj.bias':
                nn.init.zeros_(model.patch_embed.proj.bias)
                print(f"Initialized {key} with zeros.")
            elif key.startswith('transformer.'):
                # Initialize Transformer layers
                parts = key.split('.')
                layer_idx = int(parts[1])
                submodule = getattr(model.transformer, str(layer_idx))
                if 'layer_norm1.weight' in key:
                    nn.init.ones_(submodule.layer_norm1.weight)
                    print(f"Initialized {key} with ones.")
                elif 'layer_norm1.bias' in key:
                    nn.init.zeros_(submodule.layer_norm1.bias)
                    print(f"Initialized {key} with zeros.")
                elif 'layer_norm2.weight' in key:
                    nn.init.ones_(submodule.layer_norm2.weight)
                    print(f"Initialized {key} with ones.")
                elif 'layer_norm2.bias' in key:
                    nn.init.zeros_(submodule.layer_norm2.bias)
                    print(f"Initialized {key} with zeros.")
                elif 'attention.in_proj_weight' in key or 'attention.in_proj_bias' in key:
                    # Initialize attention weights if missing
                    if 'weight' in key:
                        nn.init.xavier_uniform_(submodule.attention.in_proj_weight)
                        print(f"Initialized {key} with Xavier uniform.")
                    else:
                        nn.init.zeros_(submodule.attention.in_proj_bias)
                        print(f"Initialized {key} with zeros.")
                elif 'attention.out_proj.weight' in key or 'attention.out_proj.bias' in key:
                    if 'weight' in key:
                        nn.init.xavier_uniform_(submodule.attention.out_proj.weight)
                        print(f"Initialized {key} with Xavier uniform.")
                    else:
                        nn.init.zeros_(submodule.attention.out_proj.bias)
                        print(f"Initialized {key} with zeros.")
                elif 'mlp.0.weight' in key or 'mlp.3.weight' in key:
                    nn.init.xavier_uniform_(submodule.mlp[0].weight)
                    nn.init.xavier_uniform_(submodule.mlp[3].weight)
                    print(f"Initialized {key} with Xavier uniform.")
                elif 'mlp.0.bias' in key or 'mlp.3.bias' in key:
                    nn.init.zeros_(submodule.mlp[0].bias)
                    nn.init.zeros_(submodule.mlp[3].bias)
                    print(f"Initialized {key} with zeros.")
                else:
                    print(f"No initializer for {key}, skipping.")
            else:
                print(f"No initializer for {key}, skipping.")

    # Set model to evaluation mode
    model.eval()

    # Preprocess the input image
    image_path = "/home/aniket/EECE7368/prj-vit-team/MLP/SRC/images/apple.jpg"   # Replace with your image path
    if not os.path.exists(image_path):
        print(f"Image file not found at {image_path}")
        sys.exit(1)

    try:
        input_tensor = preprocess_image(image_path, img_size=img_size)  # (1, 3, 256, 256)
    except ValueError as ve:
        print(ve)
        sys.exit(1)
    except TypeError as te:
        print(f"TypeError during image preprocessing: {te}")
        sys.exit(1)

    # Forward pass through the model
    with torch.no_grad():
        print(f"Input tensor shape: {input_tensor.shape}")
        logits = model(input_tensor)  # (1, num_classes)

    # Get predicted class
    predicted_class = get_predicted_class(logits)
    print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()
