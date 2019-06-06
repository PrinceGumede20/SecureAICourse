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
