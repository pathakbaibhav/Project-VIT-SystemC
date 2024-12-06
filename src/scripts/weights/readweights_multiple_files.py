import sys
import os
import timm

def save_single_weight(output_dir, name, array):
    # Extract the category and block name based on the weight name
    if 'patch_embed.proj' in name:
        category = 'embedding'
    elif 'cls_token' in name:
        category = 'embedding'
    elif 'pos_embed' in name:
        category = 'embedding'
    elif 'blocks' in name:
        block_number = name.split('.')[1]
        category = f'trBlock{block_number}'
        # Remove the block prefix (e.g., 'blocks.0.' -> '')
        name = name.replace(f'blocks.{block_number}.', '')
    elif 'norm' in name:
        category = 'mlp'
    else:
        category = None
    
    if category:
        # Generate the directory path for each category (embedding, trBlockX, etc.)
        block_dir = os.path.join(output_dir, category)
        os.makedirs(block_dir, exist_ok=True)
        
        # Generate the file path for the weight file (e.g., 'attn_proj_bias_weights.txt')
        file_name = f"{name.replace('.', '_')}_weights.txt"
        file_path = os.path.join(block_dir, file_name)

        # Flatten the array and convert it to a C++ array format
        flat_array = array.flatten()
        cpp_array = ", ".join(map(str, flat_array))

        # Write to the file
        with open(file_path, "w") as f:
            f.write(f"// {name}\n")
            f.write(f"const float {name.replace('.', '_')}[] = {{ {cpp_array} }};\n\n")

        print(f"Saved {name} weights to {file_path}")

def save_weights(output_dir):
    # Using the ViT base model, trained/finetuned on ImageNet
    model_name = 'vit_base_patch16_224.orig_in21k'
    print(f"Loading weights from {model_name}")

    # Load the model
    model = timm.create_model(model_name, pretrained=True)

    # Extract its weights + architecture
    state_dict = model.state_dict()

    # Save weights to corresponding files
    for name, tensor in state_dict.items():
        print(f"Saving weight for {name}")
        save_single_weight(output_dir, name, tensor.cpu().numpy())
    
    print(f"Weights saved to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <output_file_path>")
        sys.exit(1)

    output_file_path = sys.argv[1]
    save_weights(output_file_path)
