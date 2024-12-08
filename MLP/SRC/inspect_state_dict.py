# inspect_state_dict.py

import torch

def main():
    state_dict_path = "/home/aniket/EECE7368/prj-vit-team/MLP/SRC/weights/trained_vit.pth"
    state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
    
    # Check if 'state_dict' is nested
    if 'state_dict' in state_dict:
        print("Nested 'state_dict' found. Extracting...")
        state_dict = state_dict['state_dict']
    
    print("Original State Dict Keys:")
    for key in state_dict.keys():
        print(key)
    
    if key == 'vit.embeddings.position_embeddings':
        num_patches_checkpoint = checkpoint_embedding.size(1) - 1  # Subtract 1 for [CLS] token
        num_patches_model = model_embedding.size(1) - 1  # Subtract 1 for [CLS] token

        resized_embedding = resize_positional_embeddings(value, model_embedding, num_patches_checkpoint, num_patches_model)
        new_state_dict['pos_embed.pos_embed'] = resized_embedding
        print(f"Resized positional embeddings from shape {checkpoint_embedding.size()} to {model_embedding.size()}")


    

if __name__ == "__main__":
    main()
