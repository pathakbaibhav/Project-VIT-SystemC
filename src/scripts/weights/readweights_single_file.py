import sys
import timm

def save_weights(output_path):
    # Using the ViT tiny model, trained/finetuned on imagenet
    model_name = 'vit_base_patch16_224.orig_in21k'
    print(f"Loading weights from {model_name}")

    # Load the model
    model = timm.create_model(model_name, pretrained=True)

    # Extract its weights + architecture
    state_dict = model.state_dict()

    # Create a dictionary to store the weights
    weights = {}

    for name, tensor in state_dict.items():
        print(name)
        weights[name] = tensor.cpu().numpy()

    # Save weights as c++ arrays
    with open(output_path, "w") as f:
        for name, array in weights.items():
            # Flatten the array
            flat_array = array.flatten()

            # Convert to c++ array
            cpp_array = ", ".join(map(str, flat_array))
            f.write(f"// {name}\n")
            f.write(f"const float {name.replace('.', '_')}[] = {{ {cpp_array} }};\n\n")

    print(f"Weights saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <output_file_path>")
        sys.exit(1)

    output_file_path = sys.argv[1]
    save_weights(output_file_path)
