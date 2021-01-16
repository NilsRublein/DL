# -*- coding: utf-8 -*-
"""

This file applies various adversarial attacks on images from the imagenet dataset using the torchattacks library and saves them as png files. 
In addition, we also test the attacks on the inception v3 network to see how robust the prediction accuracy is depending on the chosen attack.

Implemented attacks:
    FGSM
    BIM
    CW
    RFGSM
    PGD
    FFGSM
    TPGD
    MIFGSM
    
To install torchattacks, simply use "pip install torchattacks" for your environment.
Adopted from on:  https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb

"""

import numpy as np
import pandas as pd
from skimage import io
from PIL import Image
import json
import os
import sys
import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import torchattacks

#%% 
use_cuda = True

if use_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"{device=}")

#%% Load data NEW
class CustomDataSet():
    """
    Custom defined dataset. __getitem__ is used to retrieve one element from the dataset.
    """

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

        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = self.transform(image)

        else:
            image = self.transform(image)

        return image, self.labels[img_name]

def get_labels():
    """
    Find label for each of the present images, specified in dev_dataset.csv
    :return: dict[image_name]: label
    """
    csv_path = os.getcwd() + r"\data\dev_dataset.csv"
    df = pd.read_csv(csv_path, usecols=['ImageId', 'TrueLabel', 'TargetClass'])
    present_images = os.listdir(path + r"\imagenet")
    labels = {}
    for image in present_images:
        labels[image] = int(df.loc[df['ImageId'] == image[:-4]]['TrueLabel'].iloc[0])-1
        print(image)
        print(labels[image])
    return labels

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

path = os.getcwd() + r"\data"
image_folder = os.getcwd() + r"\data\imagenet" # This folfer only contains 5 images for testing, feel free to use a different one
dataset = CustomDataSet(root_dir=image_folder, transform=None)
labels = get_labels()
data_loader = torch.utils.data.DataLoader(dataset, shuffle=False)


class_idx = json.load(open("./data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
my_iter = iter(data_loader)

#%% Call this cell to iterate to the next image and display it
images, labels = next(my_iter)
print("True Image & True Label")
imshow(make_grid(images, normalize=True), [idx2label[i] for i in labels])


#%% Load inception v3
# This one is gonna take a few minutes 

class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std
    
# Adding a normalization layer for Inception v3.
# We can't use torch.transforms because it supports only non-batch images.
norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

model = nn.Sequential(
    norm_layer,
    models.inception_v3(pretrained=True)
).to(device)

model = model.eval()
  
#%% Adversarial Attack 

atks = [torchattacks.FGSM(model, eps=8/255),
        torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=7),
        torchattacks.CW(model, c=1, kappa=0, steps=1000, lr=0.01),
        torchattacks.RFGSM(model, eps=8/255, alpha=4/255, steps=1),
        torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7),
        torchattacks.FFGSM(model, eps=8/255, alpha=12/255),
        torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=7),
        torchattacks.MIFGSM(model, eps=8/255, decay=1.0, steps=5),
       ]

print("Adversarial Image & Predicted Label")

for atk in atks :
    
    print("-"*70)
    print(atk)
    
    correct = 0
    total = 0
    
    for images, labels in data_loader:
        
        start = time.time()
        adv_images = atk(images, labels)
        labels = labels.to(device) 
        outputs = model(adv_images)

        _, pre = torch.max(outputs.data, 1)

        total += 1
        correct += (pre == labels).sum()
        
        # Save adv img as png, include attack name in the file name
        name = [idx2label[i] for i in labels]
        attack_name = atk.attack
        save_image(make_grid(adv_images.cpu().data),r'./data/atk_images/' + str(name) + '_' + str(attack_name) + '.png')
        
        imshow(make_grid(adv_images.cpu().data, normalize=True), [idx2label[i] for i in pre])

    print('Total elapsed time (sec) : %.2f' % (time.time() - start))
    print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))

