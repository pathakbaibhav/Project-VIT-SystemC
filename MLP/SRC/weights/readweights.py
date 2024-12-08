# readweights.py

import sys
import os
import torch
import numpy as np
import torch.nn as nn
import warnings

# Suppress the FutureWarning related to torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the directory containing 'vit.py' to sys.path
sys.path.append('/home/aniket/EECE7368/prj-vit-team/MLP/SRC')

# Import the vit module
import vit

def rename_state_dict_keys(state_dict):
    """
    Renames the keys in the state_dict to match the custom VisionTransformer's expected keys.

    Args:
        state_dict (dict): Original state dictionary with keys from the saved model.

    Returns:
        dict: New state dictionary with renamed keys, excluding incompatible ones.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('vit.embeddings.'):
            # Handle embedding-related keys
            suffix = key[len('vit.embeddings.'):]  # Remove 'vit.embeddings.' prefix
            if suffix == 'cls_token':
                new_key = 'cls_token'
                new_state_dict[new_key] = value
            elif suffix == 'position_embeddings':
                # This should now align with the updated model
                new_key = 'pos_embed'
                new_state_dict[new_key] = value
            elif suffix.startswith('patch_embeddings.projection.'):
                # Map 'patch_embeddings.projection.weight' to 'patch_embed.proj.weight'
                proj_suffix = suffix[len('patch_embeddings.projection.'):]
                if proj_suffix in ['weight', 'bias']:
                    new_key = f'patch_embed.proj.{proj_suffix}'
                    new_state_dict[new_key] = value
                else:
                    print(f"Skipping unhandled projection key: {key}")
            else:
                # Unhandled embedding key, skip
                print(f"Skipping unhandled embedding key: {key}")
                continue

        elif key.startswith('vit.encoder.layer.'):
            # Handle transformer encoder layer keys
            parts = key.split('.')
            if len(parts) >= 7:
                layer_num = parts[3]
                submodule = parts[4]
                att_mlp = parts[5]
                param = parts[6]

                if submodule == 'attention' and att_mlp == 'attention':
                    if param == 'attention.query.weight':
                        new_key = f'transformer.{layer_num}.attention.in_proj_weight'
                    elif param == 'attention.query.bias':
                        new_key = f'transformer.{layer_num}.attention.in_proj_bias'
                    elif param == 'attention.output.dense.weight':
                        new_key = f'transformer.{layer_num}.attention.out_proj.weight'
                    elif param == 'attention.output.dense.bias':
                        new_key = f'transformer.{layer_num}.attention.out_proj.bias'
                    else:
                        # Unhandled attention parameter, skip
                        print(f"Skipping unhandled attention parameter: {key}")
                        continue
                elif submodule == 'intermediate':
                    if param == 'dense.weight':
                        new_key = f'transformer.{layer_num}.mlp.0.weight'
                    elif param == 'dense.bias':
                        new_key = f'transformer.{layer_num}.mlp.0.bias'
                    else:
                        print(f"Skipping unhandled intermediate parameter: {key}")
                        continue
                elif submodule == 'output':
                    if param == 'dense.weight':
                        new_key = f'transformer.{layer_num}.mlp.3.weight'
                    elif param == 'dense.bias':
                        new_key = f'transformer.{layer_num}.mlp.3.bias'
                    else:
                        print(f"Skipping unhandled output parameter: {key}")
                        continue
                elif submodule.startswith('layernorm_before'):
                    if param == 'weight':
                        new_key = f'transformer.{layer_num}.layer_norm1.weight'
                    elif param == 'bias':
                        new_key = f'transformer.{layer_num}.layer_norm1.bias'
                    else:
                        print(f"Skipping unhandled layernorm_before parameter: {key}")
                        continue
                elif submodule.startswith('layernorm_after'):
                    if param == 'weight':
                        new_key = f'transformer.{layer_num}.layer_norm2.weight'
                    elif param == 'bias':
                        new_key = f'transformer.{layer_num}.layer_norm2.bias'
                    else:
                        print(f"Skipping unhandled layernorm_after parameter: {key}")
                        continue
                else:
                    # Unhandled submodule, skip
                    print(f"Skipping unhandled submodule: {key}")
                    continue

                new_state_dict[new_key] = value
            else:
                # Key does not have enough parts, skip
                print(f"Skipping key with insufficient parts: {key}")
                continue

        elif key.startswith('vit.layernorm.'):
            # Handle final layer normalization
            parts = key.split('.')
            if len(parts) >= 3:
                param = parts[2]
                if param == 'weight':
                    new_key = 'norm.weight'
                elif param == 'bias':
                    new_key = 'norm.bias'
                else:
                    print(f"Skipping unhandled layernorm parameter: {key}")
                    continue  # Unhandled parameter, skip
                new_state_dict[new_key] = value
            else:
                print(f"Skipping layernorm key with insufficient parts: {key}")
                continue

        else:
            # Unhandled key, skip
            print(f"Skipping unhandled key: {key}")
            continue

    return new_state_dict

def load_model_weights(model_path):
    """
    Loads the model's state dictionary from a .pth file.

    Args:
        model_path (str): Path to the saved .pth file.

    Returns:
        dict: Renamed state dictionary compatible with the custom VisionTransformer.
    """
    # Load the original state dict
    state_dict = torch.load(model_path, map_location='cpu')

    # Rename the keys to match the custom model
    new_state_dict = rename_state_dict_keys(state_dict)

    # Initialize the model structure
    model = vit.VisionTransformer()

    # Load the renamed state dict into the model
    # Use strict=False to allow missing/unexpected keys
    try:
        load_result = model.load_state_dict(new_state_dict, strict=False)
    except RuntimeError as e:
        print("Error loading state_dict:", e)
        sys.exit(1)

    # Optional: Print missing and unexpected keys for debugging
    if load_result.missing_keys:
        print("Missing keys in the state_dict:", load_result.missing_keys)
    if load_result.unexpected_keys:
        print("Unexpected keys in the state_dict:", load_result.unexpected_keys)

    model.eval()
    return new_state_dict

def save_weights_to_cpp(state_dict, output_path):
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
            array = tensor.numpy().flatten()
            array_name = name.replace('.', '_') + '_weights'
            f.write(f'const float {array_name}[{array.size}] = {{\n    ')
            for i, val in enumerate(array):
                f.write(f'{val:.8f}f, ')
                if (i + 1) % 8 == 0:
                    f.write('\n    ')
            f.write('\n};\n\n')
        f.write('#endif // WEIGHTS_H\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python readweights.py <model_path.pth> <output_weights.h>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_weights_path = sys.argv[2]

    # Ensure the output directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)

    # Load the state dictionary
    state_dict = load_model_weights(model_path)
    print("Embedding is completed in patch embedding")  # Existing print statement

    # Save to C++ header
    save_weights_to_cpp(state_dict, output_weights_path)
    print(f"Weights exported to {output_weights_path}")
