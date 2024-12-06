import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Floating-Point MLP Classification Block
# ---------------------------

class MLPClassificationBlock(nn.Module):
    """
    MLP Classification Block as described in Task 3.

    Args:
        embed_dim (int): Dimension of the [CLS] token (D).
        hidden_dim (int): Dimension of the hidden layer (H).
        num_classes (int): Number of output classes (K).
        use_softmax (bool): Whether to apply softmax activation on the output logits.
    """
    def __init__(self, embed_dim, hidden_dim, num_classes, use_softmax=False):
        super(MLPClassificationBlock, self).__init__()
        self.embed_dim = embed_dim  # D
        self.hidden_dim = hidden_dim  # H
        self.num_classes = num_classes  # K
        self.use_softmax = use_softmax

        # First Linear Layer: W1 and b1
        self.fc1 = nn.Linear(embed_dim, hidden_dim)

        # Activation Function: GELU
        self.gelu = nn.GELU()

        # Second Linear Layer: W2 and b2
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        if self.use_softmax:
            self.softmax = nn.Softmax(dim=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize weights using a truncated normal distribution
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, transformer_output):
        """
        Forward pass of the MLP Classification Block.

        Args:
            transformer_output (torch.Tensor): Output from the transformer block of shape (batch_size, N+1, D)

        Returns:
            logits (torch.Tensor): Output logits of shape (batch_size, K)
        """
        # Step 1: Input Extraction - Extract [CLS] token
        # [CLS] token is at index 0
        cls_token = transformer_output[:, 0, :]  # Shape: (batch_size, D)

        # Step 2: First Linear Transformation
        z = self.fc1(cls_token)  # Shape: (batch_size, H)

        # Step 3: Activation Function
        a = self.gelu(z)  # Shape: (batch_size, H)

        # Step 4: Second Linear Transformation
        y = self.fc2(a)  # Shape: (batch_size, K)

        # Step 5: Optional Softmax Activation
        if self.use_softmax:
            y = self.softmax(y)  # Shape: (batch_size, K)

        return y

# ---------------------------
# Fixed-Point Simulation Components
# ---------------------------

class FixedPointLinear(nn.Module):
    """
    Simulated Fixed-Point Linear Layer.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        scale_in (float): Scaling factor for input.
        scale_out (float): Scaling factor for output.
    """
    def __init__(self, in_features, out_features, scale_in=128.0, scale_out=128.0):
        super(FixedPointLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale_in = scale_in
        self.scale_out = scale_out

    def forward(self, x):
        # Simulate fixed-point by scaling and rounding
        x_fixed = (x * self.scale_in).round()
        weight_fixed = (self.linear.weight * self.scale_in).round()
        bias_fixed = (self.linear.bias * self.scale_out).round()

        # Perform integer matrix multiplication
        y_fixed = F.linear(x_fixed, weight_fixed, bias_fixed)

        # Scale back to floating-point
        y = y_fixed.float() / (self.scale_in * self.scale_out)
        return y

class MLPClassificationBlockFixedPoint(nn.Module):
    """
    MLP Classification Block with Fixed-Point Simulation.

    Args:
        embed_dim (int): Dimension of the [CLS] token (D).
        hidden_dim (int): Dimension of the hidden layer (H).
        num_classes (int): Number of output classes (K).
        use_softmax (bool): Whether to apply softmax activation on the output logits.
        scale_in (float): Scaling factor for input.
        scale_hidden (float): Scaling factor for hidden layer.
        scale_out (float): Scaling factor for output layer.
    """
    def __init__(self, embed_dim, hidden_dim, num_classes, use_softmax=False, 
                 scale_in=128.0, scale_hidden=128.0, scale_out=128.0):
        super(MLPClassificationBlockFixedPoint, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_softmax = use_softmax

        self.scale_in = scale_in
        self.scale_hidden = scale_hidden
        self.scale_out = scale_out

        # Fixed-Point Linear Layers
        self.fc1_fixed = FixedPointLinear(embed_dim, hidden_dim, scale_in, scale_hidden)
        self.fc2_fixed = FixedPointLinear(hidden_dim, num_classes, scale_hidden, scale_out)

        # Activation Function: Approximate GELU using Tanh (simpler for fixed-point)
        self.activation = nn.Tanh()  # Placeholder; replace with fixed-point compatible activation if needed

        if self.use_softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, transformer_output):
        # Step 1: Input Extraction - Extract [CLS] token
        cls_token = transformer_output[:, 0, :]  # Shape: (batch_size, D)

        # Step 2: First Fixed-Point Linear Transformation
        z = self.fc1_fixed(cls_token)  # Shape: (batch_size, H)

        # Step 3: Activation Function (simulated)
        a = self.activation(z)  # Shape: (batch_size, H)

        # Step 4: Second Fixed-Point Linear Transformation
        y = self.fc2_fixed(a)  # Shape: (batch_size, K)

        # Step 5: Optional Softmax Activation
        if self.use_softmax:
            y = self.softmax(y)  # Shape: (batch_size, K)

        return y

# ---------------------------
# Example Usage
# ---------------------------

if __name__ == "__main__":
    # Define parameters
    batch_size = 2
    N = 256  # Number of patches
    D = 768  # Embedding dimension
    H = 3072  # Hidden layer size
    K = 10   # Number of classes

    # Create dummy transformer output
    # Shape: (batch_size, N+1, D)
    dummy_transformer_output = torch.randn(batch_size, N + 1, D)

    # ---------------------------
    # Floating-Point MLP Classification Block
    # ---------------------------
    print("=== Floating-Point MLP Classification Block ===")
    mlp_classifier_fp = MLPClassificationBlock(embed_dim=D, hidden_dim=H, num_classes=K, use_softmax=False)
    logits_fp = mlp_classifier_fp(dummy_transformer_output)
    print("Logits shape (Floating-Point):", logits_fp.shape)  # Expected: (batch_size, K)
    print("Logits (Floating-Point):", logits_fp)

    # With Softmax
    mlp_classifier_fp_softmax = MLPClassificationBlock(embed_dim=D, hidden_dim=H, num_classes=K, use_softmax=True)
    logits_fp_softmax = mlp_classifier_fp_softmax(dummy_transformer_output)
    print("\nLogits with Softmax shape (Floating-Point):", logits_fp_softmax.shape)  # Expected: (batch_size, K)
    print("Probabilities (Floating-Point):", logits_fp_softmax)

    # ---------------------------
    # Fixed-Point MLP Classification Block
    # ---------------------------
    print("\n=== Fixed-Point MLP Classification Block ===")
    mlp_classifier_fp_fixed = MLPClassificationBlockFixedPoint(
        embed_dim=D, 
        hidden_dim=H, 
        num_classes=K, 
        use_softmax=False,
        scale_in=128.0, 
        scale_hidden=128.0, 
        scale_out=128.0
    )
    logits_fixed = mlp_classifier_fp_fixed(dummy_transformer_output)
    print("Logits shape (Fixed-Point):", logits_fixed.shape)  # Expected: (batch_size, K)
    print("Logits (Fixed-Point):", logits_fixed)

    # With Softmax
    mlp_classifier_fp_fixed_softmax = MLPClassificationBlockFixedPoint(
        embed_dim=D, 
        hidden_dim=H, 
        num_classes=K, 
        use_softmax=True,
        scale_in=128.0, 
        scale_hidden=128.0, 
        scale_out=128.0
    )
    logits_fixed_softmax = mlp_classifier_fp_fixed_softmax(dummy_transformer_output)
    print("\nLogits with Softmax shape (Fixed-Point):", logits_fixed_softmax.shape)  # Expected: (batch_size, K)
    print("Probabilities (Fixed-Point):", logits_fixed_softmax)
