import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Splits the image into patches and embeds them.
    """
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Using a Conv2d layer to perform patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, img_size, img_size)
        Returns:
            patches: (batch_size, num_patches, embed_dim)
        """
        x = self.proj(x)  # (batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2))
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
    def forward(self, x):
        """
        Args:
            x: (sequence_length, batch_size, embed_dim)
        Returns:
            out: (sequence_length, batch_size, embed_dim)
        """
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output

class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder layer.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.msa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Multi-Head Self-Attention
        x_residual = x
        x = self.norm1(x)
        # Transpose for PyTorch's MultiheadAttention: (seq, batch, embed)
        x = x.transpose(0, 1)
        attn_output = self.msa(x)
        attn_output = attn_output.transpose(0, 1)
        x = x_residual + self.dropout(attn_output)
        
        # MLP
        x_residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x_residual + self.dropout(x)
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.
    """
    def __init__(self, 
                 img_size=256, 
                 patch_size=16, 
                 in_channels=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 mlp_dim=3072, 
                 dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)
        
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, img_size, img_size)
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Concatenate class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, 1 + num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed  # Broadcasting
        x = self.dropout(x)
        
        # Transformer Encoder
        for layer in self.encoder:
            x = layer(x)
        
        # Classification Head
        x = self.norm(x)
        cls_token_final = x[:, 0]  # (batch_size, embed_dim)
        logits = self.head(cls_token_final)  # (batch_size, num_classes)
        return logits

# Example Usage
if __name__ == "__main__":
    # Define model parameters
    img_size = 256
    patch_size = 16
    in_channels = 3
    num_classes = 10  # Example for 10 classes
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_dim = 3072
    dropout = 0.1

    # Instantiate the model
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout
    )

    # Print model architecture
    print(model)

    # Create a dummy input tensor
    dummy_input = torch.randn(2, in_channels, img_size, img_size)  # Batch size of 2

    # Forward pass
    logits = model(dummy_input)
    print("Logits shape:", logits.shape)  # Expected: (2, num_classes)
