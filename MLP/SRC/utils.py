# utils.py

import torch
import vit  # Ensure vit.py is in the same directory or adjust the path accordingly

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
                print(f"Mapped {key} to {new_key}")
            elif suffix == 'position_embeddings':
                # Map to 'pos_embed.pos_embed' as per vit.py's PositionalEmbedding
                new_key = 'pos_embed.pos_embed'
                new_state_dict[new_key] = value
                print(f"Mapped {key} to {new_key}")
            elif suffix.startswith('patch_embeddings.projection.'):
                # Map 'patch_embeddings.projection.weight' to 'patch_embed.proj.weight'
                proj_suffix = suffix[len('patch_embeddings.projection.'):]
                if proj_suffix in ['weight', 'bias']:
                    new_key = f'patch_embed.proj.{proj_suffix}'
                    new_state_dict[new_key] = value
                    print(f"Mapped {key} to {new_key}")
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
                layer_num = parts[3]  # e.g., '0', '1', etc.
                submodule = parts[4]  # e.g., 'attention', 'intermediate', 'output', etc.
                att_mlp = parts[5]  # e.g., 'attention', 'layernorm_before', etc.
                param = parts[6]  # e.g., 'weight', 'bias', etc.

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
                print(f"Mapped {key} to {new_key}")
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
                print(f"Mapped {key} to {new_key}")
            else:
                print(f"Skipping layernorm key with insufficient parts: {key}")
                continue

        else:
            # Unhandled key, skip
            print(f"Skipping unhandled key: {key}")
            continue

    print(f"Total keys in new_state_dict: {len(new_state_dict)}")
    return new_state_dict
