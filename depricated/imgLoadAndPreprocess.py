import os
import random
import sys
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """Loads an image, resizes it to 224x224, and returns the image as a numpy array."""
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image to 256x256
    img = np.array(img)  # Convert image to numpy array
    return img

def load_random_image(image_dir):
    """Randomly selects and loads an image from the directory."""
    # Get all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.JPEG'))]
    
    # Randomly select an image
    random_image_file = random.choice(image_files)
    
    # Extract the label part from the filename
    label = random_image_file.split('_', 1)[1].split('.')[0]
    
    # Load and preprocess the image
    image_path = os.path.join(image_dir, random_image_file)
    image = preprocess_image(image_path)
    
    return image, label

def write_image_to_csv(image, label, output_file):
    """Writes the image data as a CSV file."""
    with open(output_file, "w") as f:
        f.write(f"label,{label}\n")  # Write the label
        f.write("r,g,b\n")  # Write header for RGB columns
        for row in image:
            for pixel in row:
                f.write(f"{pixel[0]},{pixel[1]},{pixel[2]}\n")

def main(image_dir, output_file):
    image, label = load_random_image(image_dir)

    # Write the preprocessed image to a CSV file
    write_image_to_csv(image, label, output_file)

    print(f"Label: {label}")
    print(f"Preprocessed Image Shape: {image.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_directory> <output_file>")
        sys.exit(1)

    image_dir = sys.argv[1]
    output_file = sys.argv[2]
    main(image_dir, output_file)
