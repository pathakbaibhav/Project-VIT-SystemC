import timm

# Using the ViT tiny model, trained/finetuned on imagenet
model = timm.create_model('vit_tiny_patch16_384.augreg_in21k_ft_in1k', pretrained=True)

# Save the model weights
state_dict = model.state_dict()

# Create a dictionary to store converted weights
weights = {}

for name, tensor in state_dict.items():
    weights[name] = tensor.cpu().numpy()

# Save weights as c++ arrays
with open("weights.cpp", "w") as f:
    for name, array in weights.items():
        # Flatten the array
        flat_array = array.flatten()

        # Convert to c++ array
        cpp_array = ", ".join(map(str, flat_array))
        f.write(f"// {name}\n")
        f.write(f"const float {name.replace('.', '_')}[] = {{ {cpp_array} }};\n\n")

print("FP16 weights saved to weights.cpp")