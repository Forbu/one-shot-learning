"""NTM Read/Write Head."""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class NTMHeadBase(nn.Module):
    """An NTM Read/Write Head."""

    def __init__(self, memory, controller_size,gamma=0.9):
        """Initilize the read/write head.
        :param memory: The :class:`NTMMemory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super(NTMHeadBase, self).__init__()

        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size
        
        self.w_u = torch.ones(self.N)*1/self.N
        self.gamma = gamma
        

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError

    def is_read_head(self):
        return NotImplementedError

    def _address_memory(self, k, β, g, s, γ, w_prev):
        # Handle Activations
        return NotImplementedError

class NTMReadHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMReadHead, self).__init__(memory, controller_size)

        # Corresponding to k, K sizes from the paper
        self.read_lengths = [self.M, 1]
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        # The state holds the previous time step address weightings
        return Variable(torch.zeros(batch_size, self.N))

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform(self.fc_read.weight, gain=1.4)
        nn.init.normal(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, embeddings):
        """NTMReadHead forward function.
        :param embeddings: input representation of the controller.
        """
        o = self.fc_read(embeddings)
        k, β = _split_cols(o, self.read_lengths)

        # Read from memory
        w = self._address_memory_read(k, β)
        r = self.memory.read(w)

        return r, w
    
    def _address_memory_read(self, k, β):
        # Handle Activations
        k = k.clone()
        β = F.softplus(β)

        w = self.memory.address_read(k, β)
        
        return w


class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMWriteHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ, e, a sizes from the paper
        self.write_lengths = [self.M, 1]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        return Variable(torch.zeros(batch_size, self.N))

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform(self.fc_write.weight, gain=1.4)
        nn.init.normal(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, embeddings, w_r_prev, w_u):
        """NTMWriteHead forward function.
        :param embeddings: input representation of the controller.
        :param w_r_prev: previous step state of read weigth
        :param w_u: weigth usage 
        """
        o = self.fc_write(embeddings)
        k, β = _split_cols(o, self.write_lengths)

        # Write to memory
        w = self._address_memory_write(β, w_r_prev, w_u)
        self.memory.write(w, k)

        return w
    
    def _address_memory_write(self, alpha, w_r_prev, w_u):
        # Handle Activations

        alpha = F.sigmoid(alpha)
        w_lu = Variable(torch.zeros(self.N))

        value_min, index = torch.min(w_u[0].data, 1)
        
        w_lu[index] = 1
        
        w = self.memory.address_write(alpha, w_r_prev, w_lu)
        
        return w
