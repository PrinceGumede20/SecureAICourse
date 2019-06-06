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
