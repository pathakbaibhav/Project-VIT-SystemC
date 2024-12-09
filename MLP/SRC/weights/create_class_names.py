import json
import urllib.request

# Download the ImageNet class index file
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
class_index_path = "imagenet_class_index.json"
urllib.request.urlretrieve(url, class_index_path)

# Load the class index
with open(class_index_path) as f:
    class_idx = json.load(f)

# Create class_names.txt
with open("class_names.txt", "w") as f:
    for idx in range(len(class_idx)):
        class_name = class_idx[str(idx)][1]
        f.write(f"{class_name}\n")

print("class_names.txt has been created with ImageNet class labels.")
