# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:42:59 2018

@author: 2622792
"""

"""
Testing
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
from os import listdir
import argparse

from torch.autograd import Variable


def one_hot_v2(batch,depth):
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0,batch)

# We import the dataset omniglot
directory_main = "./data/images_background"

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

n_images = len(data)
print(n_images)
#%%    
import torch
import torch.nn as nn

def get_episode(data_repo):
    
    indexs = list(map(lambda x :x[0],data_repo))
    max_indexs = max(indexs)
    
    index_choosen = np.random.randint(1,max_indexs,15)
    data_repo_selected = list(filter(lambda x: x[0] in index_choosen,data_repo))
    
    data_image_selected = []
    for data_index in data_repo_selected:
        im = torch.from_numpy(imageio.imread(directory_main + '/' + data_index[1]))
        data_image_selected.append((im,data_index[0]))
        
    random.shuffle(data_image_selected)
    data_image_selected = data_image_selected[:100]
    
    # now we change the label
    images = list(map(lambda x: x[0].view(1,1,105,105).float(),data_image_selected))
    
    labels = list(map(lambda x: x[1],data_image_selected))
    labels = list(map(lambda x: list(index_choosen).index(x),labels))
    
    images = torch.cat(images)
    images = images / 255
    
    # convertion of label towards hot auto encoder
    labels = torch.LongTensor(labels)

    lables_hot = one_hot_v2(labels,params['number_of_classes'])

    return Variable(images),Variable(lables_hot)

params = {}

params['number_of_classes'] = 15
params['input_controller_size'] = 128
params['controller_output_size'] = 10  
params['controller_layer_size'] = 1
params['num_heads'] = 1
params['N'] = 64
params['M'] = 100


model = torch.load("modelweight/NTM_model92000.pt")
length_episode = 100
n_episode = 1
for episode in range(n_episode):

    print("episode numero : ",episode)
    X,Y = get_episode(data)
    
    # memory matrix to null
    model.NTM_layer.init_sequence(1)
    
    y_out = Variable(torch.zeros(Y.size()))
    
    # first initialisation
    print(X[0,:,:,:].unsqueeze(0))
    null_var = Variable(torch.FloatTensor(1,params['number_of_classes']).zero_())
    null_var[0,0] = 1
    
    y_out[0] = model(X[0,:,:,:].unsqueeze(0),null_var)
    
    for i in range(1,length_episode):
        
        y_out[i,:] = model(X[i,:,:,:].unsqueeze(0),Y[i-1,:].unsqueeze(0))
        






