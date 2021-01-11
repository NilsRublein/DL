"""
Modified from: https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/get_started_pytorch.py

The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using e.g. FGSM. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
(The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.)


To DO:
    - hyperparameter optimization, what parameters producde the best attacks with least amount of visible pertubance?
    - check norm values for PGD and WS
    - Display images w/ pertubance, only pertubance
    - Do we want to use a simple net like this? We can also use a more sophisticated, pretrained model, e.g. Resnet50
    - Fix WS bug (if we want to use WS)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# import tensorflow as tf

from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, Wasserstein
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist


#%% Step 0: Define the neural network model, return logits instead of activation in forward method


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

#%% Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

#%% Step 4: Train the ART classifier
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

#%% Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

#%% Function for attacking and evaluating accuracy 

def eval_attack(attack):
    # Step 6: Generate adversarial test examples
    x_test_adv = attack.generate(x=x_test)
    
    # Step 7: Evaluate the ART classifier on adversarial test examples  
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples with: {}%".format(accuracy * 100))
    return accuracy

#%% FGSM
FGSM = FastGradientMethod(estimator=classifier, eps=0.2)
acc_FGSM = eval_attack(FGSM)

#%% BIM (basic iterative method, also called iterative FGSM)
BIM = BasicIterativeMethod(estimator=classifier, eps=0.2)
acc_BIM = eval_attack(BIM)

#%% PGD (Projected Gradient Descent )
# this is the "madry et al. attack" described in the NIPS competition paper
# for the oroginal paper, see https://adversarial-robustness-toolbox.readthedocs.io/en/stable/modules/attacks/evasion.html#projected-gradient-descent-pgd-pytorch

PGD = ProjectedGradientDescent(estimator=classifier, eps=0.2)
acc_PGD = eval_attack(PGD)


#%% Wasserstein, one of the recenst attacks (via Projected Sinkhorn Iterations)
# see this paper: https://arxiv.org/abs/1902.07906
WS = Wasserstein(estimator=classifier, eps=0.2)
acc_WS = eval_attack(WS)


#%% Random resizing and padding Defense
# Implementation of the paper below but in PyTorch
# https://github.com/cihangxie/NIPS2017_adv_challenge_defense
# https://arxiv.org/pdf/1711.01991.pdf

# apply original function and new on the same imagine (without randaomization), should give the same shape ?


# Takes resized image as inut and adds random padding
# def padding_layer_iyswim(inputs, shape, name=None):
#     h_start = shape[0]
#     w_start = shape[1]
#     output_short = shape[2]
#     input_shape = torch.tensor(inputs).size()
#
#     input_short = torch.min(input_shape[1:3]) # get smallest dimension ?
#
#     input_long = torch.max(input_shape[1:3]) # get longest dimension ?
#
#     output_long = torch.int32(torch.ceil(1. * torch.float(output_short) * torch.float(input_long) / torch.float(input_short)))
#
#     output_height = torch.int32(input_shape[1] >= input_shape[2]) * output_long + torch.int32(input_shape[1] < input_shape[2]) * output_short
#
#     output_width = torch.int32(input_shape[1] >= input_shape[2]) * output_short + torch.int32(input_shape[1] < input_shape[2]) * output_long
#
#     return tf.pad(inputs, torch.int32(torch.stack([[0, 0], [h_start, output_height - h_start - input_shape[1]], [w_start, output_width - w_start - input_shape[2]], [0, 0]])), name=name)


"""
resize_shape_ = np.random.randint(310, 331)

shape_tensor: np.array([random.randint(0, FLAGS.image_resize - resize_shape_), random.randint(0, FLAGS.image_resize - resize_shape_), FLAGS.image_resize])

"""


#%% tf padding

# t = tf.constant([[1, 2, 3], [4, 5, 6]])
# paddings = tf.constant([[1, 1,], [2, 2]])
# # 'constant_values' is 0.
# # rank of 't' is 2.
# tf_pad = tf.pad(t, paddings, "CONSTANT")
# print("tf padding:\n", tf_pad)

#%% pytorch padding

# import torch.nn.functional as F
# t = torch.tensor([[1, 2, 3], [4, 5, 6]])
# pad_ = (1,1,2,2)
# pytorch_pad = F.pad(input=t, pad=pad_, mode='constant', value=0)

# print("pytorch padding:\n", pytorch_pad)

"""
t = torch.Tensor([[1,2,3],[4,5,6]])
pad = (1,1,2,2)  # Should be TBLR. Now it is LRTB

def pad_from_tf_to_pt(padding):
  return (pad[2], pad[3], pad[0], pad[1])

pad = pad_from_tf_to_pt(pad)
pytorch_pad = F.pad(input=t, pad=pad, mode='constant', value=0)
print("Pytorch padding:\n", pytorch_pad)
"""

