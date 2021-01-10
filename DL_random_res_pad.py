<<<<<<< HEAD
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
=======
# -*- coding: utf-8 -*-
"""
Random resizing and padding as input for a NN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#from art.utils import load_mnist



#%% Get a random rescaling factor

resize_shape_ = np.random.randint(299, 331)  # Check this values
resize_factor = resize_shape  /299
#%% Step 0: Define the neural network model, return logits instead of activation in forward method

class Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, input_, shape, rand):
        super(Net, self).__init__()
        
        resize =  nn.Upsample(scale_factor=val, mode='nearest') #Can also look into different modes like bilinear (in the paper itself they used nearest)
        res_img = resize(input_)
            #torch.nn.Upsample, or 
            #torchvision.transforms.Resize
        
        #pad
            # torchvision.transforms.Pad
        
        #flatten
        
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

#%% Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Swap axes to PyTorch's NCHW format

x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

#%% Step 2: Create the model

model = Net()

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 20
accuracies = []

#%% Train and eval 


for epoch in range(1, epochs+1):
    for i, (minibatch_data, minibatch_label) in enumerate(trainloader):
        minibatch_data, minibatch_label = minibatch_data.to(device), minibatch_label.to(device)  # Move the data to the device that is used

        # Set dL/dU, dL/dV, dL/dW to be filled with zeros
        optimizer.zero_grad()

        # forward the minibatch through the net  
        prob = model(minibatch_data.view(-1, w * h * c)) # this bit fltattens the data, we dont want that here
        
        # Compute the average of the losses of the data points in the minibatch
        loss = criterion(prob, minibatch_label) 

        # backward pass to compute dL/dU, dL/dV and dL/dW    
        loss.backward()
        
        # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
        optimizer.step()

    # Evaluate the model on the test set
    correct_total = 0
    for i, (x_batch, y_batch) in enumerate(testloader):
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move the data to the device that is used

      y_pred = model(x_batch.view(-1, w * h * c))
      y_pred_max = torch.argmax(y_pred, dim=1)

      correct_total += torch.sum(torch.eq(y_pred_max, y_batch)).item()

    accuracy = correct_total / len(testset.data)
    accuracies.append(accuracy*100)

    print(f'Epoch #{epoch} ({optimiser_method}) - Test accuracy: {correct_total / len(testset.data):.3f}')

  return accuracies



>>>>>>> e599b41369af597c0b8e08f0727580c18b62b574
