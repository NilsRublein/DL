# -*- coding: utf-8 -*-
"""
Random resizing and padding as input for a NN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

# Cannot be installed through Pipenv. Use pip install --no-deps torchvision instead.
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from sklearn.model_selection import train_test_split
from art.utils import load_mnist


#%%
use_cuda = True

if use_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"{device=}")
#%% Step 0: Define the neural network model, return logits instead of activation in forward method

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Resize
        # Pad
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


def get_random_padding(max_padding):
    max_padding = 30
    pad_left = torch.randint(high=max_padding, size=(1, 1)).item()
    pad_right = max_padding - pad_left
    pad_top = torch.randint(high=max_padding, size=(1, 1)).item()
    pad_bottom = max_padding - pad_top
    LRTB = (pad_left, pad_right, pad_top, pad_bottom)
    return LRTB


#%% Step 1: Load the MNIST dataset

def run(use_padding_and_scaling=False, use_MNIST=False):
    if use_padding_and_scaling:
        transform = transforms.Compose([
            transforms.Resize(size=1.0, interpolation=nn.Bilinear),
            torchvision.transforms.Pad(padding=get_random_padding(max_padding=30)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    if use_MNIST:
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

        # Step 1a: Swap axes to PyTorch's NCHW format
        x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
        x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
    else:
        # https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2
        dataset = datasets.ImageFolder(path, transform=transform)
        x_train, y_train, x_test, y_test = train_test_split(dataset, test_size=0.1)
        trainloader = torch.utils.data.DataLoader(train_data,
                                                  batch_size=32)

    #%% Step 2: Create the model
    model = Net()

    # Step 2a: Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)


if __name__ == "__main__":
    path = os.getcwd() + r"\data"
    run(use_padding_and_scaling=False)
