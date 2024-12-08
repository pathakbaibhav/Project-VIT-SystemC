# vit.py

import torch
import torch.nn as nn
import math
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        """
        Args:
            img_size (int): Size of the input image (assumed square).
            patch_size (int): Size of each patch (assumed square).
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            embed_dim (int): Dimension of the embedding space.
        """
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional layer for patch embedding
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        print("Embedding is completed in patch embedding")

    def forward(self, x):
        x = self.proj(x)  # (B, D, H_patch, W_patch)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D) where N = H_patch * W_patch
        B, N, D = x.shape
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return self.pos_embed

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        """
        Transformer Encoder Block consisting of:
        - Layer Normalization
        - Multi-head Self-Attention
        - Residual Connection
        - MLP with GELU activation and Dropout
        """
        super(TransformerEncoderBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Forward pass for the Transformer Encoder Block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N+1, D)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, N+1, D)
        """
        # Layer normalization before attention
        x_norm = self.layer_norm1(x)
        # Self-attention expects input shape as (N, B, D)
        attn_output, _ = self.attention(x_norm.transpose(0, 1), x_norm.transpose(0, 1), x_norm.transpose(0, 1))
        attn_output = attn_output.transpose(0, 1)  # (B, N+1, D)
        x = x + attn_output  # Residual connection

        # Layer normalization before MLP
        x_norm = self.layer_norm2(x)
        mlp_output = self.mlp(x_norm)  # (B, N+1, D)
        x = x + mlp_output  # Residual connection

        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768,
                 num_heads=12, mlp_dim=3072, num_layers=12, num_classes=1000, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder layers
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize [CLS] token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes)
        """
        print(f"Input shape: {x.shape}")
        x = self.patch_embed(x)  # (B, N, D)
        print(f"After patch_embed shape: {x.shape}")
        B, N, D = x.shape

        # Prepare the [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # Concatenate [CLS] token with patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)

        # Add positional embeddings
        pos_embed = self.pos_embed(x)  # (1, N+1, D)
        print(f"x shape before adding pos_embed: {x.shape}")  # Debug
        print(f"pos_embed shape: {pos_embed.shape}")  # Debug

        x = x + pos_embed  # Broadcast positional embeddings
        x = self.dropout(x)

        # Pass through the transformer encoder
        x = self.transformer(x)  # (B, N+1, D)

        # Apply layer normalization
        x = self.norm(x)  # (B, N+1, D)

        # Extract the [CLS] token representation
        cls_token_final = x[:, 0]  # (B, D)

        # Classification head
        logits = self.classifier(cls_token_final)  # (B, num_classes)
        return logits

# The MLPClassificationHead class has been removed as it's not used in VisionTransformer

def preprocess_image(image_path, img_size=256):
    """
    Loads and preprocesses an image.

    Args:
        image_path (str): Path to the image file.
        img_size (int): Size to resize the image to (assumed square).
    
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, img_size, img_size)
    """
    import cv2

    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")
    print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")

    # Convert BGR to RGB
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Converted to RGB: shape: {img.shape}, dtype: {img.dtype}")
    except cv2.error as e:
        raise ValueError(f"Error converting image to RGB: {e}")

    # Resize image
    try:
        img = cv2.resize(img, (img_size, img_size))
        print(f"Resized image: shape: {img.shape}, dtype: {img.dtype}")
    except cv2.error as e:
        raise ValueError(f"Error resizing image: {e}")

    # Normalize to [0,1]
    try:
        img = img.astype(np.float32) / 255.0
        print(f"Normalized image: shape: {img.shape}, dtype: {img.dtype}")
    except Exception as e:
        raise ValueError(f"Error normalizing image: {e}")

    # Ensure the image has 3 channels
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image with 3 channels, got shape {img.shape}")

    # Convert to tensor and add batch dimension
    try:
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        print(f"Converted to tensor: shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
    except Exception as e:
        raise TypeError(f"Error converting image to tensor: {e}")
    
    return img_tensor
