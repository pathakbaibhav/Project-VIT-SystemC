import torch
from transformers import ViTForImageClassification
import numpy as np
import sys
import os

def extract_weights(model_name: str, output_pth: str, output_h: str):
    """
    Loads a pretrained ViT model, extracts its weights, and saves them to a .pth file and a C++ header file.
    
    Args:
        model_name (str): Hugging Face model identifier.
        output_pth (str): Path to save the PyTorch state dictionary.
        output_h (str): Path to save the C++ header file.
    """
    # Load the pretrained ViT model
    print(f"Loading model '{model_name}' from Hugging Face...")
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully.")

    # Extract state dictionary
    print("Extracting state dictionary...")
    state_dict = model.state_dict()
    print("State dictionary extracted.")

    # Save the state dictionary to a .pth file
    print(f"Saving state dictionary to '{output_pth}'...")
    torch.save(state_dict, output_pth)
    print(f"State dictionary saved to '{output_pth}'.")

    # Save the weights to a C++ header file
    print(f"Converting weights to C++ header file '{output_h}'...")
    save_weights_to_cpp(state_dict, output_h)
    print(f"Weights exported to '{output_h}'.")

def save_weights_to_cpp(state_dict: dict, output_path: str):
    """
    Saves the state dictionary to a C++ header file as constant float arrays.

    Args:
        state_dict (dict): PyTorch state dictionary.
        output_path (str): Path to the output C++ header file.
    """
    with open(output_path, 'w') as f:
        f.write('// Auto-generated weights.h\n\n')
        f.write('#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n')
        for name, tensor in state_dict.items():
            # Replace '.' with '_' to conform to C++ variable naming
            array_name = name.replace('.', '_')
            # Convert tensor to numpy array and flatten
            array = tensor.cpu().numpy().flatten()
            # Determine array size
            array_size = array.size
            # Write the array in C++ syntax
            f.write(f'const float {array_name}[{array_size}] = {{\n    ')
            for i, val in enumerate(array):
                f.write(f'{val:.8f}f, ')
                # Insert a newline every 8 elements for readability
                if (i + 1) % 8 == 0:
                    f.write('\n    ')
            f.write('\n};\n\n')
        f.write('#endif // WEIGHTS_H\n')

if __name__ == "__main__":
    # Define your specific paths here
    model_name = 'google/vit-base-patch16-224'
    output_pth = '/home/aniket/EECE7368/prj-vit-team/MLP/SRC/weights/trained_vit.pth'
    output_h = '/home/aniket/EECE7368/prj-vit-team/MLP/SRC/weights/weights.h'

    # Ensure the output directories exist
    os.makedirs(os.path.dirname(output_pth), exist_ok=True)
    os.makedirs(os.path.dirname(output_h), exist_ok=True)

    extract_weights(model_name, output_pth, output_h)
