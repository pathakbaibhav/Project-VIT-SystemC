# utils.py

import torch
import vit  # Ensure vit.py is in the same directory or adjust the path accordingly

# utils.py

import re

def rename_state_dict_keys(state_dict):
    """
    Renames keys from the checkpoint to match the model's state_dict.

    Args:
        state_dict (dict): Original state_dict from the checkpoint.

    Returns:
        dict: Renamed state_dict compatible with the model.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Mapping for embedding layers
        if key == 'vit.embeddings.cls_token':
            new_key = 'cls_token'
        elif key == 'vit.embeddings.position_embeddings':
            new_key = 'pos_embed.pos_embed'
        elif key == 'vit.embeddings.patch_embeddings.projection.weight':
            new_key = 'patch_embed.proj.weight'
        elif key == 'vit.embeddings.patch_embeddings.projection.bias':
            new_key = 'patch_embed.proj.bias'
        
        # Mapping for transformer encoder layers
        elif key.startswith('vit.encoder.layer.'):
            # Example key: 'vit.encoder.layer.0.attention.attention.query.weight'
            pattern = r'^vit\.encoder\.layer\.(\d+)\.attention\.attention\.(query|key|value)\.(weight|bias)$'
            match = re.match(pattern, key)
            if match:
                layer_num, attn_type, param = match.groups()
                new_key = f"transformer.{layer_num}.attention.{attn_type}.{param}"
            else:
                # Handle other attention-related keys
                pattern_out_proj = r'^vit\.encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)$'
                match_out = re.match(pattern_out_proj, key)
                if match_out:
                    layer_num, param = match_out.groups()
                    new_key = f"transformer.{layer_num}.attention.out_proj.{param}"
                else:
                    # Handle intermediate dense layers
                    pattern_intermediate = r'^vit\.encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)$'
                    match_intermediate = re.match(pattern_intermediate, key)
                    if match_intermediate:
                        layer_num, param = match_intermediate.groups()
                        new_key = f"transformer.{layer_num}.mlp.0.{param}"
                    else:
                        # Handle output dense layers
                        pattern_output = r'^vit\.encoder\.layer\.(\d+)\.output\.dense\.(weight|bias)$'
                        match_output = re.match(pattern_output, key)
                        if match_output:
                            layer_num, param = match_output.groups()
                            new_key = f"transformer.{layer_num}.mlp.3.{param}"
                        else:
                            # Handle layernorm_before and layernorm_after
                            pattern_ln_before = r'^vit\.encoder\.layer\.(\d+)\.layernorm_before\.(weight|bias)$'
                            match_ln_before = re.match(pattern_ln_before, key)
                            if match_ln_before:
                                layer_num, param = match_ln_before.groups()
                                new_key = f"transformer.{layer_num}.layer_norm1.{param}"
                            else:
                                pattern_ln_after = r'^vit\.encoder\.layer\.(\d+)\.layernorm_after\.(weight|bias)$'
                                match_ln_after = re.match(pattern_ln_after, key)
                                if match_ln_after:
                                    layer_num, param = match_ln_after.groups()
                                    new_key = f"transformer.{layer_num}.layer_norm2.{param}"
                                else:
                                    # If none of the patterns match, skip the key
                                    print(f"Skipping unhandled key: {key}")
                                    continue

        # Mapping for final layer normalization
        elif key == 'vit.layernorm.weight':
            new_key = 'norm.weight'
        elif key == 'vit.layernorm.bias':
            new_key = 'norm.bias'

        # Mapping for classifier
        elif key == 'vit.classifier.weight':
            new_key = 'classifier.weight'
        elif key == 'vit.classifier.bias':
            new_key = 'classifier.bias'
        
        else:
            print(f"Skipping unhandled key: {key}")
            continue  # Skip keys that are not handled
        
        # Assign the value to the new key
        new_state_dict[new_key] = value

    return new_state_dict
