# Weights

Included here is a script `readweights.py` that takes the weights from the pretrained `vit_tiny_patch16_384.augreg_in21k_ft_in1k` model and writes them to a file as c++ arrays. When using the standard float datatype, this file ends up being over 70MB so its too large for github. 

In order to use this script, install the `timm` python package using `pip install timm`. This package contains the model that we need along with other PyTorch vision models if we want to experiment.
