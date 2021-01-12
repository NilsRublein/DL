# -*- coding: utf-8 -*-
"""
Random resizing and padding as input for a NN
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# Cannot be installed through Pipenv. Use pip install --no-deps torchvision instead.
import torchvision
import torchvision.transforms as transforms
from skimage import io  # , transform
from torch.utils.data import Dataset, DataLoader

from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, Wasserstein
from art.estimators.classification import PyTorchClassifier
from attack import AttackWrapper

from PIL import Image
# %%
use_cuda = True

if use_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"{device=}")

import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

# %% Step 0: Define the neural network model, return logits instead of activation in forward method

# NIPS defense.gi py: https://github.com/cihangxie/NIPS2017_adv_challenge_defense/blob/master/defense.py

min_resize = 310
max_resize = 331
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CustomDataSet(Dataset):
    """
    Custom defined dataset. __getitem__ is used to retrieve one element from the dataset.
    """

    def __init__(self, root_dir, use_padding_and_scaling):
        self.root_dir = root_dir
        self.use_padding_and_scaling = use_padding_and_scaling
        self.all_imgs = os.listdir(root_dir)
        self.labels = get_labels()
        self.count = 0

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = self.all_imgs[index]
        image = io.imread(self.root_dir + "\\" + img_name)

        new_size = get_resize(min_resize, max_resize)

        if self.use_padding_and_scaling:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=new_size),  # Default: nearest bilinear
                torchvision.transforms.Pad(padding=get_random_padding(new_size, max_resize)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        image = self.transform(image)
        # plt.imshow(image.permute(1, 2, 0))
        # plt.show()

        image = self.transform(image)

        return image, self.labels[img_name]


def get_random_padding(new_size, max_resize):
    """
    Insert random amount on padding on each side.
    pad_left + pad_right = max_padding
    pad_top + pad_bottom = max_padding
    :return tuple
    """
    max_pad = max_resize - new_size

    pad_left = np.random.randint(0, max_pad)
    pad_right = max_pad - pad_left

    pad_top = np.random.randint(0, max_pad)
    pad_bottom = max_pad - pad_top

    LRTB = (pad_left, pad_top, pad_right, pad_bottom)
    return LRTB


def get_labels():
    """
    Find label for each of the present images, specified in dev_dataset.csv
    :return: dict[image_name]: label
    """
    csv_path = os.getcwd() + r"\data\dev_dataset.csv"
    df = pd.read_csv(csv_path, usecols=['ImageId', 'TrueLabel', 'TargetClass'])
    present_images = os.listdir(path + r"\images")
    labels = {}
    for image in present_images:
        labels[image] = int(df.loc[df['ImageId'] == image[:-4]]['TrueLabel'].iloc[0])-1
    return labels


def get_resize(min=299, max=331):
    resize_shape = np.random.randint(min, max)  # Check this values
    return resize_shape


def show_sample(dataset, labels, n=4):
    fig = plt.figure()

    for i in range(len(dataset)):
        image = dataset[i].permute(1, 2, 0)

        ax = plt.subplot(1, n, i + 1)
        plt.tight_layout()
        label = labels[i].item()
        label_text = classes[label]

        try:
            comma = label_text.index(',')
            label_text = label_text[:comma]
        except ValueError:
            pass

        ax.set_title(f'{label_text} ({label})')
        ax.axis('off')
        plt.imshow(image.cpu())
        if i == len(dataset)-1:
            plt.show()
            break


# %% Step 1: Load the MNIST dataset

def run(use_padding_and_scaling, batch_size):

    image_folder = os.getcwd() + r"\data\images"
    labels = get_labels()

    dataset = CustomDataSet(root_dir=image_folder, use_padding_and_scaling=use_padding_and_scaling)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Class definitions (indices 0-1000)
    with open(os.getcwd() + r'\data\labels.txt') as json_file:
        labels = json.load(json_file)
        global classes
        classes = {int(key): val for key, val in labels.items()}

    # %% Step 2: Create the model
    # model = Net(num_classes=1000)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
    model = torchvision.models.inception_v3(pretrained=True, progress=True, transform_input=True).to(device)

    # Step 2a: Define the loss function and the optimizerTrue
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #%% Train the model
    start = time.time()
    # for epoch in range(0, 5):
    #
    #     plt.ion()
    #     losses = []
    #     loss_plot = plt.plot(0, 0)[0]
    #     model.train()  # Put the network in train mode
    #     for i, (x_batch, y_batch) in enumerate(trainloader):
    #         x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move the data to the device that is used
    #         # show_sample(x_batch, y_batch)
    #         optimizer.zero_grad()  # Set all currently stored gradients to zero
    #         y_pred = model(x_batch)
    #         loss = criterion(y_pred, y_batch)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Compute relevant metrics
    #         y_pred_max = torch.argmax(y_pred, dim=1)  # Get the labels with highest output probability
    #         correct = torch.sum(torch.eq(y_pred_max, y_batch)).item()  # Count how many are equal to the true labels
    #         elapsed = time.time() - start  # Keep track of how much time has elapsed
    #
    #         losses.append(loss.item())
    #         loss_plot.set_xdata(len(losses))
    #         loss_plot.set_ydata(losses)
    #         plt.draw()
    #         plt.pause(0.01)
    #
    #         # Show progress every 20 batches
    #         if not i % 20:
    #             print(
    #                 f'epoch: {epoch}, time: {elapsed:.3f}s, loss: {loss.item():.3f}, train accuracy: {correct / 1:.5f}')
    #

    if False:
        correct_total = 0
        total_tested = 0
        model.eval()  # Put the network in eval mode
        for i, (x_batch, y_batch) in enumerate(testloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move the data to the device that is used
            # show_sample(x_batch, y_batch)
            y_pred = model(x_batch)
            y_pred_max = torch.argmax(y_pred, dim=1)

            total_tested += len(y_batch)
            correct_total += torch.sum(torch.eq(y_pred_max, y_batch)).item()
            print(f"Correct: {correct_total}/{total_tested} ({correct_total/total_tested*100:.2f}%)")

        print(f'Accuracy on the test set: {correct_total / len(test_dataset):.5f}')

    #%% Attack section

    all_imgs_loaded = []
    for img in train_dataset.dataset.all_imgs:
        all_imgs_loaded.append(np.asarray(Image.open(f"{path}\\images\\{img}")))

    attack = AttackWrapper(model, all_imgs_loaded, criterion, optimizer, min_resize, max_resize)
    FGSM = FastGradientMethod(estimator=attack.classifier, eps=0.2)

    acc_FGSM = attack.eval_attack(FGSM)
    print(f"FGSM Accuracy: {acc_FGSM}")


if __name__ == "__main__":
    path = os.getcwd() + r"\data"
    run(use_padding_and_scaling=True, batch_size=4)
