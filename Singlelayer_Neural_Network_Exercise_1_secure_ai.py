# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:42:07 2019

@author: Prince
"""

import torch

def activation(x):
    return 1/(1+torch.exp(-x))

#Generates random data
torch.manual_seed(7)

features =torch.randn((1, 5))

weights =torch.randn_like(features)

bias = torch.randn((1, 1))

Output = activation(torch.sum(features * weights) + bias)

print(Output)

## Calculate the output of this network using matrix multiplication Exercise Two

mytensor1 = features.shape
mytensor2 = weights.shape

newweights =weights.view(5 , 1)
myTensor = activation(torch.mm(features, weights.view(5 , 1)) + bias)
print(myTensor)

#Calculate the multilayer Network

torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     
n_hidden = 2                     
n_output = 1                    

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)
print(output)
