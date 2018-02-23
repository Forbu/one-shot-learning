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

from NTM.NTM_fullmodel import EncapsulatedNTM

from NTM_LRU.NTM_fullmodel import EncapsulatedNTM_LRU

class One_shot_classifier(nn.Module):
    def __init__(self,n_output,controller_size,controller_layers,num_heads,N,M):
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
                nn.MaxPool2d(4),
                nn.Conv2d(16, 32, kernel_size=3, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(4),
                nn.Conv2d(32, 32, kernel_size=3, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        
        # get the dimension output
        """
        NTM modified layer for one shot learning :
        """
        num_inputs = 576 + n_output
        
        self.NTM_layer = EncapsulatedNTM(num_inputs, n_output,
                 controller_size, controller_layers, num_heads, N, M)
    
    def reset_parameters(self):
        
        for module in self.representation_layer.children():
            classname = module.__class__.__name__
            if classname.find('Conv2d') != -1:
                module.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm2d') != -1:
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.fill_(0)
                
    def forward(self, images_t,label_t_1):
        # Input 105x105
        represent = self.representation_layer(images_t)
        
        # transforming into 32 feature map of 8x8
        represent = represent.view(-1, 576)
        
        # Aggregation of the representation layer and the label information
        aggregation = torch.cat((represent, label_t_1),1)

        result, previous_state = self.NTM_layer(aggregation)
        return result

class One_shot_classifier_LRU(nn.Module):
    def __init__(self,n_output,controller_size,controller_layers,num_heads,N,M):
        super(One_shot_classifier_LRU, self).__init__()
        """
        Three components :
            - Convolutionnal layer
            - Neural Turing machine layer
        """
        self.representation_layer = nn.Sequential(
                nn.Conv2d(1,16,kernel_size=3, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(4),
                nn.Conv2d(16, 32, kernel_size=3, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(4),
                nn.Conv2d(32, 32, kernel_size=3, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        
        self.reset_parameters()
        # get the dimension output
        """
        NTM modified layer for one shot learning :
        """
        num_inputs = 576 + n_output
        
        self.NTM_layer = EncapsulatedNTM_LRU(num_inputs, n_output,
                 controller_size, controller_layers, num_heads, N, M)

    def reset_parameters(self):
        
        for module in self.representation_layer.children():
            
            classname = module.__class__.__name__

            
            if classname.find('Conv2d') != -1:

                module.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm2d') != -1:
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.fill_(0)
        
    def forward(self, images_t,label_t_1):
        # Input 105x105
        represent = self.representation_layer(images_t)
        
        # transforming into 32 feature map of 8x8
        represent = represent.view(-1, 576)
        
        # Aggregation of the representation layer and the label information        
        aggregation = torch.cat((represent, label_t_1),1)

        result, previous_state = self.NTM_layer(aggregation)
        return result


class One_shot_classifier_reduce(nn.Module):
    def __init__(self,n_output,controller_size,controller_layers,num_heads,N,M):
        super(One_shot_classifier_reduce, self).__init__()

        # get the dimension output
        """
        NTM modified layer for one shot learning :
        """
        num_inputs = 400 + n_output
        
        self.NTM_layer = EncapsulatedNTM_LRU(num_inputs, n_output,
                 controller_size, controller_layers, num_heads, N, M)

        
    def forward(self, images_t,label_t_1):
        # transforming into 32 feature map of 8x8
        images_t = images_t.view(-1, 400)
        
        # Aggregation of the representation layer and the label information 

        aggregation = torch.cat((images_t, label_t_1),1)

        result, previous_state = self.NTM_layer(aggregation)
        return result


 