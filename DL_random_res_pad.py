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
import pandas as pd
from skimage import io, transform
import json
import time

# Cannot be installed through Pipenv. Use pip install --no-deps torchvision instead.
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
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
        self.conv_1 = nn.Conv2d(in_channels=299*299, out_channels=4, kernel_size=5, stride=1)
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


class CustomDataSet(Dataset):
    '''
    Custom defined dataset. __getitem__ is used to retrieve one element from the dataset.
    '''

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.all_imgs = os.listdir(root_dir)
        self.labels = get_labels()

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = self.all_imgs[index]
        image = io.imread(self.root_dir + "\\" + img_name)

        if self.transform:
            image = self.transform(image)
        return image, self.labels[img_name]


def get_random_padding(max_padding=30):
    '''
    Insert random amount on padding on each side.
    pad_left + pad_right = max_padding
    pad_top + pad_bottom = max_padding
    :return tuple
    '''
    pad_left = int(torch.randint(high=max_padding, size=(1, 1)).item())
    pad_right = max_padding - pad_left
    pad_top = int(torch.randint(high=max_padding, size=(1, 1)).item())
    pad_bottom = max_padding - pad_top
    LRTB = (pad_left, pad_right, pad_top, pad_bottom)
    return LRTB


def get_labels():
    '''
    Find label for each of the present images, specified in dev_dataset.csv
    :return: dict[image_name]: label
    '''
    csv_path = os.getcwd() + r"\data\dev_dataset.csv"
    df = pd.read_csv(csv_path, usecols=['ImageId', 'TrueLabel', 'TargetClass'])
    present_images = os.listdir(path + r"\images")
    labels = {}
    for image in present_images:
        labels[image] = int(df.loc[df['ImageId'] == image[:-4]]['TrueLabel'].iloc[0])
    return labels


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

        # Swap axes to PyTorch's NCHW format
        x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
        x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
    else:
        image_folder = os.getcwd() + r"\data\images"
        dataset = CustomDataSet(root_dir=image_folder, transform=transform)

        labels = get_labels()

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        # Class definitions (indices 0-1000)
        with open(os.getcwd() + r'\data\labels.txt') as json_file:
            labels = json.load(json_file)
            classes = {int(key): val for key, val in labels.items()}

    #%% Step 2: Create the model
    model = Net()

    # Step 2a: Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    '''
    #%% Train the model
    start = time.time()

    for epoch in range(0, 5):

        model.train()  # Put the network in train mode
        for i, (x_batch, y_batch) in enumerate(trainloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move the data to the device that is used

            optimizer.zero_grad()  # Set all currenly stored gradients to zero
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            # Compute relevant metrics
            y_pred_max = torch.argmax(y_pred, dim=1)  # Get the labels with highest output probability
            correct = torch.sum(torch.eq(y_pred_max, y_batch)).item()  # Count how many are equal to the true labels
            elapsed = time.time() - start  # Keep track of how much time has elapsed

            # Show progress every 20 batches
            if not i % 20:
                print(
                    f'epoch: {epoch}, time: {elapsed:.3f}s, loss: {loss.item():.3f}, train accuracy: {correct / 1:.3f}')

            correct_total = 0

        model.eval()  # Put the network in eval mode
        for i, (x_batch, y_batch) in enumerate(testloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move the data to the device that is used

            y_pred = model(x_batch)
            y_pred_max = torch.argmax(y_pred, dim=1)

            correct_total += torch.sum(torch.eq(y_pred_max, y_batch)).item()

        print(f'Accuracy on the test set: {correct_total / len(test_dataset):.3f}')
    '''


if __name__ == "__main__":
    path = os.getcwd() + r"\data"
    run(use_padding_and_scaling=False)
