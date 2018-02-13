import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class NTM(nn.Module):
    """A Neural Turing Machine."""
    def __init__(self, num_inputs, num_outputs, controller, memory, heads, gamma=0.9):
        """Initialize the NTM.
        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`LSTMController`
        :param memory: :class:`NTMMemory`
        :param heads: list of :class:`NTMReadHead` or :class:`NTMWriteHead`
        Note: This design allows the flexibility of using any number of read and
              write heads independently, also, the order by which the heads are
              called in controlled by the user (order in list)
        """
        super(NTM, self).__init__()

        # Save arguments
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.memory = memory
        self.heads = heads

        self.gamma = gamma

        self.N, self.M = memory.size()
        _, self.controller_size = controller.size()

        # Initialize the initial previous read values to random biases
        self.num_read_heads = 0
        self.init_r = []
        self.init_w_u = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = Variable(torch.randn(1, self.M) * 0.01)
                self.register_buffer("read{}_bias".format(self.num_read_heads), init_r_bias.data)
                self.init_r += [init_r_bias]
                init_w_u = Variable(torch.ones(self.N)*1/self.N)
            
                self.init_w_u += [init_w_u]
                self.num_read_heads += 1

        assert self.num_read_heads > 0, "heads list must contain at least a single read head"

        # Initialize a fully connected layer to produce the actual output:
        #   [controller_output; previous_reads ] -> output
        self.fc = nn.Linear(self.controller_size + self.num_read_heads * self.M, num_outputs)
        self.reset_parameters()
        
    def create_new_state(self, batch_size):
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        init_w_u = [w_u.clone().repeat(batch_size, 1) for w_u in self.init_w_u]
        controller_state = self.controller.create_new_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]


        return init_r, controller_state, heads_state, init_w_u

    def reset_parameters(self):
        # Initialize the linear layer
        nn.init.xavier_uniform(self.fc.weight, gain=1)
        nn.init.normal(self.fc.bias, std=0.01)

    def forward(self, x, prev_state):
        """
        NTM forward function.
        :param x: input vector (batch_size x num_inputs)
        :param prev_state: The previous state of the NTM information about w_u
        """
        # Unpack the previous state
        prev_reads, prev_controller_state, prev_heads_states, prev_w_u = prev_state

        # Use the controller to get an embeddings
        inp = torch.cat([x] + prev_reads, dim=1)
        controller_outp, controller_state = self.controller(inp, prev_controller_state)

        # Read/Write from the list of heads
        reads = []
        heads_states = []
        weight_use = [w_u for w_u in prev_w_u]
        
        i = 0 
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            
            if head.is_read_head():
                prev_read_weight = prev_head_state
                r, head_state = head(controller_outp)
                reads += [r]
                
                weight_use[i] = self.gamma * weight_use[i] + head_state
                
            else:
                head_state = head(controller_outp, prev_read_weight, prev_w_u)
                weight_use[i] += head_state
                i = i + 1
                
            heads_states += [head_state]

        # Generate Output
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        o = F.log_softmax(self.fc(inp2), dim=1)

        # Pack the current state
        state = (reads, controller_state, heads_states, weight_use)

        return o, state