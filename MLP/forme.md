To find out all the branches on GitHub and pull all of them, you can follow these steps:

1. **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Fetch all branches**:
    ```bash
    git fetch --all
    ```

3. **List all branches**:
    ```bash
    git branch -r
    ```

4. **Pull all branches**:
    ```bash
    for branch in $(git branch -r | grep -v '\->'); do
        git branch --track ${branch#origin/} $branch
    done
    git fetch --all
    git pull --all
    ```

Replace `<repository-url>` with the URL of your GitHub repository and `<repository-directory>` with the name of the directory where the repository will be cloned.


  origin/HEAD -> origin/main
  origin/feedback
  origin/luyue
  origin/main
  origin/ml_aniket



  Libaries installed 
  pip install torch torchvision opencv-python numpy
pip install transformers torch torchvision
conda install -c huggingface transformers


python readweights.py /home/aniket/EECE7368/prj-vit-team/MLP/SRC/weights/trained_vit.pth /home/aniket/EECE7368/prj-vit-team/MLP/SRC/weights/weights.h