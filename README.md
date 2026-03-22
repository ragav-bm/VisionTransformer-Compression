# CIFAR-10 Classification with Vision Transformers

This repository contains a project for training and evaluating neural networks, including Vision Transformers (ViT) and Cross Vision Transformers (CrossViT), on the CIFAR-10 dataset. The project is implemented in Python using PyTorch.

## Folder Structure
├── main.py # Main script for training and testing models 
├── my_models.py # Implementation of ViT and CrossViT models


## Requirements

- Python 3.8 or higher
- PyTorch
- torchvision
- einops
- argparse

Install the required Python packages using:

bash
pip install torch torchvision einops
Usage
Training a Model
Run the main.py script to train a model on the CIFAR-10 dataset. You can specify various arguments to customize the training process.
python [main.py](http://_vscodecontentref_/3) --model cvit --batch-size 64 --epochs 10 --lr 0.001 --aug

Arguments
--model: Model to train (r18, vit, or cvit). Default is cvit.
--batch-size: Batch size for training. Default is 64.
--epochs: Number of epochs to train. Default is 5.
--lr: Learning rate. Default is 0.003.
--momentum: Momentum for SGD optimizer. Default is 0.9.
--no-cuda: Disable CUDA training.
--seed: Random seed. Default is 1.
--log-interval: Number of batches to wait before logging training status. Default is 10.
--save-model: Save the trained model.
--dry-run: Perform a single pass for debugging.
--aug: Enable data augmentation.
Models
ResNet-18 (r18): A standard ResNet-18 model.
Vision Transformer (vit): A simple Vision Transformer implementation.
Cross Vision Transformer (cvit): A multi-scale Vision Transformer with cross-attention.
Example
To train the CrossViT model with data augmentation for 10 epochs:

python [main.py](http://_vscodecontentref_/4) --model cvit --epochs 10 --aug

Workspace
(rerun without)
Collecting workspace information

Here is a README.md file for your GitHub repository:

. ├── commit-with-timestamps.sh # Script for committing files with their modification timestamps ├── main.py # Main script for training and testing models ├── my_models.py # Implementation of ViT and CrossViT models

Usage
Training a Model
Run the main.py script to train a model on the CIFAR-10 dataset. You can specify various arguments to customize the training process.

Arguments
--model: Model to train (r18, vit, or cvit). Default is cvit.
--batch-size: Batch size for training. Default is 64.
--epochs: Number of epochs to train. Default is 5.
--lr: Learning rate. Default is 0.003.
--momentum: Momentum for SGD optimizer. Default is 0.9.
--no-cuda: Disable CUDA training.
--seed: Random seed. Default is 1.
--log-interval: Number of batches to wait before logging training status. Default is 10.
--save-model: Save the trained model.
--dry-run: Perform a single pass for debugging.
--aug: Enable data augmentation.
Models
ResNet-18 (r18): A standard ResNet-18 model.
Vision Transformer (vit): A simple Vision Transformer implementation.
Cross Vision Transformer (cvit): A multi-scale Vision Transformer with cross-attention.
Example
To train the CrossViT model with data augmentation for 10 epochs:

Saving Models
Trained models are saved in the current directory with the following names:

r18_model_aug.pth for ResNet-18
vit_model_aug.pth for ViT
Cvit_model_aug.pth for CrossViT

File Descriptions
main.py: Contains the training and testing pipeline for the models.
my_models.py: Implements the Vision Transformer (ViT) and Cross Vision Transformer (CrossViT) architectures.
Dataset
The CIFAR-10 dataset is automatically downloaded and stored in the specified folder. It consists of 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 test images.

References
Vision Transformer (ViT)
CrossViT
License
This project is licensed under the MIT License. See the LICENSE file for details.
