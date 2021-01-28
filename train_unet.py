# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:46:38 2021

@author: Nils
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
from PIL import Image
import json
import os
import sys
import time
import re

from unet import UNet
import torch
import torch.nn as nn
import torchvision.utils
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

# Choose which device to use
use_cuda = True

if use_cuda and torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
device

#%% Load data

class CustomDataSet():
    """
    Custom defined dataset. __getitem__ is used to retrieve one element from the dataset.
    """

    def __init__(self, adv_root_dir, true_root_dir, transform=None):
        self.adv_root_dir = adv_root_dir
        self.true_root_dir = true_root_dir
        self.transform = transform
        
        self.all_adv_imgs = os.listdir(adv_root_dir)
        self.all_true_imgs = os.listdir(true_root_dir)
        
        self.labels = get_labels()
        
            
    def __len__(self):
        return len(self.all_adv_imgs)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        adv_img_name = self.all_adv_imgs[index]

        true_img_name = get_true_img_name(adv_img_name)
        
        adv_img = io.imread(self.adv_root_dir + "\\" + adv_img_name)
        true_img = io.imread(self.true_root_dir + "\\" + true_img_name)

        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
                     
            adv_img = self.transform(adv_img)
            true_img = self.transform(true_img)

        else:
            adv_img = self.transform(adv_img)
            true_img = self.transform(true_img)

        return adv_img, true_img, adv_img_name #, self.labels[adv_img_name]
    
def get_labels():
    """
    Get the label for each of the present adv images
    :return: dict[image_name]: label
    """

    present_images = os.listdir(adv_image_folder)
    labels = {}
    
    for image in present_images:
        label = str(image).partition("-")[0] # get everything in the string until it encounters a "-" -> aka the label
        labels[image] = label
    return labels

def get_true_img_name(adv_name):
    """
    Strip the name of the adv img to get the name of the originale file
    e.g. from 'Dog-12342134214.png#FGSM.png' to '12342134214.png'
    :return: str(name of original file)
    """
    
    name = re.search('-(.*)#', adv_name)     
    name = name.group(1)
    if '-' in name:
        name = name[name.rindex('-')+1:] # Start string from last '-' so 'cat-dog-name' will be 'name'. This is bc we have some names like 'toilet-paper-571521.png'
    return name

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
    

def save_the_img(denoised_imgs, name):
    '''
    Takes in a tensor image and saves it as png at the specified location
    '''
    name = name.replace('.png','') # Get rid of all the '.pngs'
    save_image(noisy_imgs, r"./data/denoised_images/denoised" +"#" + name + ".png")


#%%
adv_image_folder = os.getcwd() + r"\data\atk_test_imgs" 
true_image_folder = os.getcwd() + r"\data\imagenet_all_images" 

dataset = CustomDataSet(adv_root_dir=adv_image_folder, true_root_dir=true_image_folder, transform=None)

batch_size = 5 # Check if this is a good value
w, h, c = 299, 299, 3 # Width, height, channels
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


my_iter = iter(trainloader)

#%% Call this cell to iterate to the next image and display it

# This needs a batchsize of 1, to actually show one image, otherwise it will make a grid of batch_size elements. 
# Are there also torchvision functions to just display one image of a dataloader with a batch_size > 1?
#adv_images, true_images,label = next(my_iter)
adv_images, true_images, img_name = next(my_iter)

print("Adv Image & Label")
#imshow(make_grid(adv_images, normalize=True), labels[0])

print("True Image & Label")
#imshow(make_grid(true_images, normalize=True), labels[0])

#%% Build unet
input_shape = c # RGB
output_shape = input_shape
model = UNet(input_shape,output_shape).to(device) 

#%% Train

"""
Nice to have:
    Add lr scheduler
    tensorboard logger
    Progressbar
"""



# Mean absolute error (MAE) between each element in the input xx and target yy
criterion = nn.L1Loss() 

# What values should we use here?
lr = 0.01
epochs = 2

optimizer=torch.optim.Adam(model.parameters(), lr=lr)

print("Starting to train model")
start = time.time()

for epoch in range(1, epochs+1):
    epoch_loss = 0
    
    # For every minibatch in trainloader, get the noisy and true images
    
    for i, (noisy_imgs, true_imgs, name) in enumerate(trainloader): 
        noisy_imgs, true_imgs = noisy_imgs.to(device), true_imgs.to(device)  # Move the data to the device that is used

        # Set dL/dU, dL/dV, dL/dW to be filled with7 zeros
        optimizer.zero_grad()

        # forward the minibatch through the net  
        #denoised_imgs = model(noisy_imgs.view(-1, w * h * c)) #CHECK IF THIS IS STILL CORRECT
        denoised_imgs = model(noisy_imgs)
        
        # Compute the average of the losses of the data points in the minibatch
        #loss = criterion(denoised_imgs, true_imgs.view(-1, w * h * c)) # For AE
        loss = criterion(denoised_imgs, true_imgs) 
        epoch_loss += loss.item()

        # backward pass to compute dL/dU, dL/dV and dL/dW    
        loss.backward()
        
        # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
        optimizer.step()
        
        if epoch == epochs:
            #print("epoch ", epoch, "of total ", epochs )
            for img, n in zip(denoised_imgs, name):
                print(" Saving denoised img: ", n)
                save_the_img(img,n)
        
        
    # compute the epoch training loss
    epoch_loss = epoch_loss / len(trainloader)
    
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch, epochs, loss))

end = time.time()
print("Time elapsed in minutes: ", (end - start)/60)

#%% Save denoised images

    
#%%
