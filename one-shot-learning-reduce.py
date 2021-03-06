# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:19:59 2018

@author: 2622792
"""
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
from models_one_shot import One_shot_classifier_reduce

params = {}

params['number_of_classes'] = 15
params['input_controller_size'] = 128
params['controller_output_size'] = 200  
params['controller_layer_size'] = 1
params['num_heads'] = 4
params['N'] = 128
params['M'] = 40

model = One_shot_classifier_reduce(params['number_of_classes'],
                            params['controller_output_size'],params['controller_layer_size'],
                            params['num_heads'],params['N'],params['M'])
                            
import random 
from scipy.misc import imresize


length_episode = 100

def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)
        
def get_episode(data_repo,batch):
    
    data_all_image = torch.zeros((100,batch,20,20))
    data_all_label = torch.zeros((batch,100,15))
    for batch_index in range(batch):
        indexs = list(map(lambda x :x[0],data_repo))
        max_indexs = max(indexs)
        
        index_choosen = np.random.randint(1,max_indexs,15)
        data_repo_selected = list(filter(lambda x: x[0] in index_choosen,data_repo))
        
        data_image_selected = []
        for data_index in data_repo_selected:
            
            image_numpy = imageio.imread(directory_main + '/' + data_index[1])
            image_numpy = imresize(image_numpy,(20,20))
            
            im = torch.from_numpy(image_numpy)
            
            data_image_selected.append((im,data_index[0]))
        
        
        random.shuffle(data_image_selected)
        data_image_selected = data_image_selected[:100]
        
        # now we change the label
        images = list(map(lambda x: x[0].view(1,1,20,20).float(),data_image_selected))
        
        labels = list(map(lambda x: x[1],data_image_selected))
        labels = list(map(lambda x: list(index_choosen).index(x),labels))
        
        images = torch.cat(images)
        images = images
        images = images / 255 * 2 -1 
        # convertion of label towards hot auto encoder
        labels = torch.LongTensor(labels)
        lables_hot = one_hot_v2(labels,params['number_of_classes'])
        
        data_all_image[:,batch_index,:,:] = images
        data_all_label[batch_index,:,:] = lables_hot
    
    return Variable(data_all_image),Variable(data_all_label)

X,Y = get_episode(data,1)
#%%
# optimizer 
optimizer = optim.Adam(model.parameters(),
                             lr=1e-3)

batch = 16
# loss
criterion = nn.BCELoss()
n_episode = 100000
# training scession
for episode in range(n_episode):
    optimizer.zero_grad()
    print("episode numero : ",episode)
    X,Y = get_episode(data,batch)
    
    # memory matrix to null
    model.NTM_layer.init_sequence(batch)
    
    y_out = Variable(torch.zeros(Y.size()))
    
    # first initialisation
    null_var = Variable(torch.FloatTensor(batch,params['number_of_classes']).zero_())
    index = np.random.randint(0,15,batch)
    for p in range(batch):
        null_var[p,index[p]] = 1.0
    
    y_out[:,0,:] = model(X[0,:,:,:].unsqueeze(0),null_var)
    
    for i in range(1,length_episode):
        y_out[:,i,:] = model(X[i,:,:,:].unsqueeze(0),Y[:,i-1,:])
        
    try:
        loss = criterion(y_out[:,1:,:], Y[:,1:,:])
        loss.backward()
        clip_grads(model)
        optimizer.step()
        print(y_out)
    except:
        print("error")

    print("loss : ",loss.data)
    if (episode % 2000) == 0:
         torch.save(model,'modelweight/NTM_model-LRU-' + str(episode) + '.pt')



