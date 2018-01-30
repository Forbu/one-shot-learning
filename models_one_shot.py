"""
Neural network for the oneshot learning
"""

"""
Sequencial :
First convolutionnal layers (representation compression)
Second Fully connected LSTM (controller)
Third memory layer (for one shot learning)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from NTM_fullmodel import EncapsulatedNTM

class One_shot_classifier(nn.Module):
    def __init__(self,n_output):
        super(One_shot_classifier, self).__init__()
        """
        Three components :
            - Convolutionnal layer
            - Neural Turing machine layer
        """
        self.representation_layer = nn.Sequential(
                nn.Conv2d(1,16,kernel_size=3, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        
        # get the dimension output
        """
        NTM modified layer for one shot learning :
        """
        num_inputs = 10
        #self.NTM_layer = EncapsulatedNTM(num_inputs, n_output,
         #        controller_size=10, controller_layers=10, num_heads=1, N=10, M=10)
        
        # then LSTM controller layer
        
    def forward(self, images_t,label_t_1):
        # On this layer we can use transfert learning
        represent = self.representation_layer(images_t)
        print(represent)
        represent = represent.view(-1, 100)
        
        # Aggregation of the representation layer and the label information
        aggregation = torch.cat((represent,label_t_1))
        
        # Prediction using the NTM layer
        #return self.NTM_layer(aggregation)
        
        
        
        



