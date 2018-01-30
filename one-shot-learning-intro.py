#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Jan 24 14:10:19 2018

@author: adrienbufort
"""

"""
One-shot learning testing with transfert learning
on CIFAR dataset

First step
    -> get neural transfert from classic dataset (cifar 10 or 10)
    -> get the data until fully connected layer
    -> add the memory layer (the one shot layer)
    -> train some part of the dataset (exemple 5 images)

"""

# import the CIFAR dataset
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
from os import listdir
import argparse

from torch.autograd import Variable

# We import the dataset omniglot
directory_main = "/Users/adrienbufort/Downloads/omniglot-master/python/images_background_small1"

# we create the dataset :
list_alphabect = listdir(directory_main)
if ".DS_Store" in list_alphabect:
    list_alphabect.remove(".DS_Store")
sub_alphabet = list(map(lambda alphabet: directory_main + "/" + alphabet,list_alphabect))
print("List of alphabet : ",sub_alphabet)

# list of label / directory
data = []
index_label = 0
    
for index,alphabet in enumerate(sub_alphabet):
    list_character = listdir(alphabet)
    
    # Now we create the list of 
    if ".DS_Store" in list_character:
        list_character.remove(".DS_Store")
    sub_character = list(map(lambda character: alphabet + "/" + character,list_character))
    
    print(sub_character)
    
    for index_2,image_repo in enumerate(sub_character):
        for image in listdir(image_repo):
            data.append((index_label,list_alphabect[index] + "/" + list_character[index_2] + "/" + image))
        index_label += 1
    
print(data)

# load and look one image :
import imageio

im = imageio.imread('/Users/adrienbufort/Downloads/omniglot-master/python/images_background/Balinese/character05/0112_05.png')
print(im.shape)

# dataset of 20000 image in pytorch tensor
data_image = []
for data_index in data:
    im = torch.from_numpy(imageio.imread(directory_main + '/' + data_index[1]))
    data_image.append((im,data_index[0]))
    
# 20 000 image 
#%% transforming the dataset into a continuous one with data transform
example_image = data_image[0][0].float()
example_image = example_image / 255
example_image = example_image.view(1,105,105)
example_image = example_image.unsqueeze(0)
print("example : ",example_image)

#%%
from models_one_shot import One_shot_classifier
model = One_shot_classifier(n_output=10)
result = model(example_image,0)












