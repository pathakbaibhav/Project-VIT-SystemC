# main.py

import torch
from vit import VisionTransformer
import torch.nn as nn
import sys
import os
import warnings
import torch.nn.functional as F
import math
from utils import rename_state_dict_keys  # Ensure utils.py is in the same directory
from PIL import Image
import torchvision.transforms as transforms
import argparse
import csv
import traceback
import matplotlib.pyplot as plt

# Suppress the FutureWarning related to torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

def preprocess_image(image_path, img_size):
    """
    Loads an image, converts it to RGB, resizes, normalizes, and converts it to a tensor.

    Args:
        image_path (str): Path to the input image.
        img_size (int): Desired image size for resizing.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, img_size, img_size)
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet means
                             std=[0.229, 0.224, 0.225])   # Standard ImageNet stds
    ])

    image = preprocess(image)  # (3, img_size, img_size)
    image = image.unsqueeze(0)  # (1, 3, img_size, img_size)

    return image

def resize_positional_embeddings(checkpoint_embedding, num_patches_model, embed_dim):
    """
    Resize positional embeddings to match the model's expected shape.

    Args:
        checkpoint_embedding (torch.Tensor): Positional embeddings from the checkpoint.
        num_patches_model (int): Number of patches in the model.
        embed_dim (int): Embedding dimension.

    Returns:
        torch.Tensor: Resized positional embeddings.
    """
    num_patches_checkpoint = checkpoint_embedding.size(1) - 1
    cls_token = checkpoint_embedding[:, :1, :]
    patch_embeddings = checkpoint_embedding[:, 1:, :]

    if num_patches_checkpoint == num_patches_model:
        return checkpoint_embedding

    h_checkpoint = int(math.sqrt(num_patches_checkpoint))
    h_model = int(math.sqrt(num_patches_model))

    if h_checkpoint ** 2 != num_patches_checkpoint or h_model ** 2 != num_patches_model:
        raise ValueError("Number of patches must be a perfect square.")

    # Reshape for interpolation
    patch_embeddings = patch_embeddings.view(1, h_checkpoint, h_checkpoint, embed_dim)
    patch_embeddings = patch_embeddings.permute(0, 3, 1, 2)  # (1, D, H_checkpoint, W_checkpoint)

    # Interpolate to new size
    resized_patch_embeddings = F.interpolate(
        patch_embeddings, size=(h_model, h_model), mode='bilinear', align_corners=False
    )

    # Reshape back
    resized_patch_embeddings = resized_patch_embeddings.permute(0, 2, 3, 1).view(1, num_patches_model, embed_dim)

    # Combine [CLS] token with resized patch embeddings
    resized_embeddings = torch.cat((cls_token, resized_patch_embeddings), dim=1)  # (1, N+1, D)

    return resized_embeddings

def get_predicted_classes(logits, top_k=1):
    """
    Determines the top-K predicted classes from logits.

    Args:
        logits (torch.Tensor): Logits tensor of shape (B, num_classes)
        top_k (int): Number of top predictions to return.

    Returns:
        list of tuples: Each tuple contains (class_index, score).
    """
    probabilities = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=-1)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    return list(zip(top_indices, top_probs))

def load_class_names(class_names_path):
    """
    Loads class names from a file.

    Args:
        class_names_path (str): Path to the class names file.

    Returns:
        list: List of class names.
    """
    if not os.path.exists(class_names_path):
        print(f"Class names file not found at {class_names_path}")
        return None

    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return class_names

def extract_ground_truth(filename):
    """
    Extracts the ground truth label from the filename.
    Assumes that the filename starts with the label (e.g., 'apple.jpg' -> 'apple').

    Args:
        filename (str): Image filename.

    Returns:
        str: Ground truth label.
    """
    base = os.path.basename(filename)
    label = os.path.splitext(base)[0]
    return label

def plot_accuracy(top1_acc, topk_acc, total):
    """
    Plots the Top-1 and Top-K accuracy.

    Args:
        top1_acc (float): Top-1 accuracy.
        topk_acc (float): Top-K accuracy.
        total (int): Total number of samples.
    """
    labels = ['Top-1 Accuracy', 'Top-K Accuracy']
    accuracies = [top1_acc * 100, topk_acc * 100]

    plt.figure(figsize=(8,6))
    bars = plt.bar(labels, accuracies, color=['skyblue', 'salmon'])
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Prediction Accuracy on {total} Images')

    # Adding the percentage labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}%', ha='center', va='bottom')

    plt.show()

def main():
    # Configuration based on your pre-trained model
    img_size = 256
    patch_size = 16
    embed_dim = 768
    num_heads = 12
    mlp_dim = 3072
    num_layers = 12
    num_classes = 15  # Adjust based on your dataset

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Vision Transformer Multi-Image Classification")
    parser.add_argument('--images_dir', type=str, required=True, help='Path to the directory containing images')
    parser.add_argument('--class_names', type=str, default=None, help='Path to the class names file (optional)')
    parser.add_argument('--output_csv', type=str, default='predictions.csv', help='Path to save predictions CSV')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top predictions to return')
    args = parser.parse_args()

    images_dir = args.images_dir
    class_names_path = args.class_names
    output_csv = args.output_csv
    top_k = args.top_k

    # Validate images directory
    if not os.path.exists(images_dir):
        print(f"Images directory not found at {images_dir}")
        sys.exit(1)
    
    # Load class names if provided
    class_names = None
    if class_names_path:
        class_names = load_class_names(class_names_path)
        if class_names is None:
            print("Proceeding without class names. Predictions will be class indices.")

    # Initialize the Vision Transformer
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.1
    )

    # Load custom weights
    state_dict_path = "/home/aniket/EECE7368/prj-vit-team/MLP/SRC/weights/trained_vit.pth"
    if not os.path.exists(state_dict_path):
        print(f"State dict file not found at {state_dict_path}")
        sys.exit(1)

    try:
        state_dict = torch.load(state_dict_path, map_location='cpu')  # Removed weights_only=True if not needed
    except Exception as e:
        print(f"Error loading state dict: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Handle nested state_dict if present
    if 'state_dict' in state_dict:
        print("Nested 'state_dict' found. Extracting...")
        state_dict = state_dict['state_dict']
    elif 'model_state_dict' in state_dict:
        print("Nested 'model_state_dict' found. Extracting...")
        state_dict = state_dict['model_state_dict']
    else:
        print("No nested state_dict found. Using the loaded state_dict directly.")

    try:
        new_state_dict = rename_state_dict_keys(state_dict)
    except Exception as e:
        print(f"Error renaming state dict keys: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Adjust positional embeddings during resizing if necessary
    if 'pos_embed.pos_embed' in new_state_dict:
        print("Resizing positional embeddings...")
        checkpoint_embedding = new_state_dict['pos_embed.pos_embed']
        num_patches_model = (img_size // patch_size) ** 2
        try:
            resized_pos_embed = resize_positional_embeddings(checkpoint_embedding, num_patches_model, embed_dim)
            new_state_dict['pos_embed.pos_embed'] = resized_pos_embed
            print(f"Resized positional embeddings to shape {resized_pos_embed.size()}")
        except Exception as e:
            print(f"Error resizing positional embeddings: {e}")
            traceback.print_exc()
            sys.exit(1)

    print(f"Total keys in new_state_dict: {len(new_state_dict)}")

    if new_state_dict is None or len(new_state_dict) == 0:
        print("Failed to rename state dict keys or new_state_dict is empty.")
        sys.exit(1)

    try:
        load_result = model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"Error loading state dict into model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Optional: Initialize missing keys if any
    if load_result.missing_keys:
        print("Missing keys:", load_result.missing_keys)
        for key in load_result.missing_keys:
            try:
                if key == 'norm.weight':
                    nn.init.ones_(model.norm.weight)
                    print(f"Initialized {key} with ones.")
                elif key == 'norm.bias':
                    nn.init.zeros_(model.norm.bias)
                    print(f"Initialized {key} with zeros.")
                elif key == 'patch_embed.proj.weight':
                    # Initialize Conv2d weights if missing
                    nn.init.kaiming_normal_(model.patch_embed.proj.weight, mode='fan_out', nonlinearity='relu')
                    print(f"Initialized {key} with Kaiming normal.")
                elif key == 'patch_embed.proj.bias':
                    nn.init.zeros_(model.patch_embed.proj.bias)
                    print(f"Initialized {key} with zeros.")
                elif key.startswith('transformer.'):
                    # Initialize transformer layers as needed
                    parts = key.split('.')
                    layer_idx = int(parts[1])
                    sub_key = '.'.join(parts[2:])

                    # Access the specific transformer layer
                    transformer_layer = model.transformer[layer_idx]

                    if sub_key == 'attention.in_proj_weight':
                        # Initialize in_proj_weight with Xavier uniform
                        nn.init.xavier_uniform_(transformer_layer.attention.in_proj_weight)
                        print(f"Initialized {key} with Xavier uniform.")
                    elif sub_key == 'attention.in_proj_bias':
                        nn.init.zeros_(transformer_layer.attention.in_proj_bias)
                        print(f"Initialized {key} with zeros.")
                    elif sub_key == 'attention.out_proj.weight':
                        nn.init.xavier_uniform_(transformer_layer.attention.out_proj.weight)
                        print(f"Initialized {key} with Xavier uniform.")
                    elif sub_key == 'attention.out_proj.bias':
                        nn.init.zeros_(transformer_layer.attention.out_proj.bias)
                        print(f"Initialized {key} with zeros.")
                    elif sub_key == 'layer_norm1.weight':
                        nn.init.ones_(transformer_layer.layer_norm1.weight)
                        print(f"Initialized {key} with ones.")
                    elif sub_key == 'layer_norm1.bias':
                        nn.init.zeros_(transformer_layer.layer_norm1.bias)
                        print(f"Initialized {key} with zeros.")
                    elif sub_key == 'layer_norm2.weight':
                        nn.init.ones_(transformer_layer.layer_norm2.weight)
                        print(f"Initialized {key} with ones.")
                    elif sub_key == 'layer_norm2.bias':
                        nn.init.zeros_(transformer_layer.layer_norm2.bias)
                        print(f"Initialized {key} with zeros.")
                    elif sub_key == 'mlp.0.weight':
                        nn.init.xavier_uniform_(transformer_layer.mlp[0].weight)
                        print(f"Initialized {key} with Xavier uniform.")
                    elif sub_key == 'mlp.0.bias':
                        nn.init.zeros_(transformer_layer.mlp[0].bias)
                        print(f"Initialized {key} with zeros.")
                    elif sub_key == 'mlp.3.weight':
                        nn.init.xavier_uniform_(transformer_layer.mlp[3].weight)
                        print(f"Initialized {key} with Xavier uniform.")
                    elif sub_key == 'mlp.3.bias':
                        nn.init.zeros_(transformer_layer.mlp[3].bias)
                        print(f"Initialized {key} with zeros.")
                    else:
                        print(f"No initializer for {key}, skipping.")
                elif key == 'classifier.weight':
                    nn.init.xavier_uniform_(model.classifier.weight)
                    print(f"Initialized {key} with Xavier uniform.")
                elif key == 'classifier.bias':
                    nn.init.zeros_(model.classifier.bias)
                    print(f"Initialized {key} with zeros.")
                else:
                    print(f"No initializer for {key}, skipping.")
            except Exception as e:
                print(f"Error initializing {key}: {e}")
                traceback.print_exc()
                continue

    # Set model to evaluation mode
    model.eval()

    # Define image extensions to process
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

    # Lists to store ground truth and predictions
    ground_truths = []
    predictions_top1 = []
    predictions_topk = []

    # Counter for correct predictions
    correct_top1 = 0
    correct_topk = 0
    total = 0

    # Prepare CSV for saving predictions
    try:
        with open(output_csv, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Create dynamic headers based on top_k
            headers = ['Image']
            for i in range(1, top_k + 1):
                headers.extend([f'Predicted Class {i}', f'Score {i}'])
            writer.writerow(headers)

            # Iterate over all images in the specified directory
            print(f"Processing images in directory: {images_dir}")
            for filename in os.listdir(images_dir):
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    image_path = os.path.join(images_dir, filename)
                    print(f"\nProcessing image: {filename}")

                    # Extract ground truth label
                    gt_label = extract_ground_truth(filename)
                    ground_truths.append(gt_label)

                    # Preprocess the image
                    try:
                        input_tensor = preprocess_image(image_path, img_size=img_size)  # (1, 3, 256, 256)
                        print(f"Input tensor type: {type(input_tensor)}, shape: {input_tensor.shape}")
                    except Exception as e:
                        print(f"Error during preprocessing: {e}")
                        traceback.print_exc()
                        writer.writerow([filename] + ['Preprocessing Error'] * (2 * top_k))
                        continue

                    # Forward pass through the model
                    try:
                        with torch.no_grad():
                            logits = model(input_tensor)  # (1, num_classes)
                            print(f"Logits type: {type(logits)}, shape: {logits.shape}")
                    except Exception as e:
                        print(f"Error during model inference: {e}")
                        traceback.print_exc()
                        writer.writerow([filename] + ['Inference Error'] * (2 * top_k))
                        continue

                    # Get top-K predicted classes
                    try:
                        topk_preds = get_predicted_classes(logits, top_k=top_k)
                        predicted_classes = [class_names[idx] if class_names else str(idx) for idx, _ in topk_preds]
                        scores = [score for _, score in topk_preds]
                        predictions_topk.append(predicted_classes)
                        predictions_top1.append(predicted_classes[0])

                        # Write to CSV
                        row = [filename]
                        for cls, score in zip(predicted_classes, scores):
                            row.extend([cls, f"{score:.4f}"])
                        writer.writerow(row)
                        print(f"Predicted Classes: {predicted_classes}, Scores: {scores}")
                    except Exception as e:
                        print(f"Error determining predicted classes: {e}")
                        traceback.print_exc()
                        writer.writerow([filename] + ['Prediction Error'] * (2 * top_k))
                        continue

                    # Update counters
                    total += 1
                    if gt_label == predicted_classes[0]:
                        correct_top1 += 1
                        correct_topk += 1
                    elif gt_label in predicted_classes:
                        correct_topk += 1

                else:
                    print(f"Skipping non-image file: {filename}")

        print(f"\nPredictions saved to {output_csv}")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Calculate accuracy
    if total > 0:
        top1_acc = correct_top1 / total
        topk_acc = correct_topk / total
        print(f"\nTotal Images Processed: {total}")
        print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")
        print(f"Top-{top_k} Accuracy: {topk_acc * 100:.2f}%")

        # Plot accuracy
        plot_accuracy(top1_acc, topk_acc, total)
    else:
        print("No images were processed.")

if __name__ == "__main__":
    main()
