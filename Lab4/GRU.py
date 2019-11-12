# @title Imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import seaborn as sns

import torch
import torch.nn as nn

from matplotlib import pyplot as plt

SEQ_LENGTH = 15  # @param {type:"integer"}
HIDDEN_UNITS = 20  # @param {type:"integer"}

dummy_input = [torch.from_numpy(np.array([[np.random.normal()]])) for _ in range(SEQ_LENGTH)]


######################
#   YOUR CODE HERE   #
######################
# Add several cell constructors (use those already defined in Tensorflow) to the
# list (e.g., also add a GRU, and a few more LSTMS with their initial
# forget_bias values set to: 0, +1, +2 and -2).
# If in doubt, check the documentation.

def _set_forget_bias(lstm_cell, fill_value=0.):
    # The bias terms in the lstm_cell are arranged as bias_input_gate, bias_forget_gate, bias_gain_gate, bias_output_gate
    # To alter the forget_gate bias, we need to modify the parameters from 1/4 to 1/2 of the length of the bias vectors
    for name, _ in lstm_cell.named_parameters():
        if "bias" in name:
            bias = getattr(lstm_cell, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(float(fill_value))

    return lstm_cell


### Solution
rnn_types = {
    'LSTM (0)': lambda nhid: _set_forget_bias(nn.modules.LSTMCell(input_size=1, hidden_size=nhid), fill_value=0.),
    ## TODO add several types of LSTM cells varying the forget gate bias - e.g. +1, -2, +2, +10
    'LSTM (+1)': lambda nhid: _set_forget_bias(nn.modules.LSTMCell(input_size=1, hidden_size=nhid), fill_value=1.),
    'LSTM (-2)': lambda nhid: _set_forget_bias(nn.modules.LSTMCell(input_size=1, hidden_size=nhid), fill_value=-2.),
    'LSTM (+2)': lambda nhid: _set_forget_bias(nn.modules.LSTMCell(input_size=1, hidden_size=nhid), fill_value=2.),
    'LSTM (+10)': lambda nhid: _set_forget_bias(nn.modules.LSTMCell(input_size=1, hidden_size=nhid), fill_value=10.),
    # add a GRUCell
    'GRU': lambda nhid: nn.modules.GRU(input_size=1, hidden_size=nhid),
    # add our RNN module
    #'RNN': lambda nhid: VanillaRNN(input_size=1, hidden_size=nhid),
}

depths = {rnn_type: [] for rnn_type in rnn_types}
grad_norms = {rnn_type: [] for rnn_type in rnn_types}

for rnn_type in rnn_types:

    # build RNN model
    constructor = rnn_types[rnn_type]
    rnn = constructor(HIDDEN_UNITS)

    # initialize gradients
    rnn.zero_grad()

    rnn_at_time = []
    gradients_at_time = []

    prev_state = None

    # pass the sequence through the RNN model
    for i in range(SEQ_LENGTH):
        ## Each RNN cell model has a different output, so switch after the defined type
        if prev_state is None:
            prev_state = rnn(dummy_input[i].float())
        else:
            if rnn_type.startswith('RNN'):
                prev_state = rnn(dummy_input[i].float(), hidden=prev_state[1])
            else:
                prev_state = rnn(dummy_input[i].float(), prev_state)

        ## We want to retain the gradient over the hidden state after each timestep (i.e. input of the sequence)
        if rnn_type.startswith('LSTM'):
            prev_state[1].retain_grad()  # for LSTMs the output is (h_t, c_t) . We call retain_grad() for c_t
            rnn_at_time.append(prev_state[1])

        ## GRUs and our RNN model have only one "hidden" output - h_t
        elif rnn_type.startswith('GRU'):
            prev_state.retain_grad()
            rnn_at_time.append(prev_state)

        elif rnn_type.startswith('RNN'):
            prev_state[1].retain_grad()
            rnn_at_time.append(prev_state[1])

    # We don't really care about the loss here: we are not solving a specific
    # problem, any loss will work to inspect the behavior of the gradient.
    dummy_loss = torch.sum(rnn_at_time[-1])
    dummy_loss.backward()

    # collect all the gradients and plot them
    for i in range(1, SEQ_LENGTH):
        current_gradient = rnn_at_time[i].grad
        gradients_at_time.append(current_gradient)

    for gid, grad in enumerate(gradients_at_time):
        depths[rnn_type].append(len(gradients_at_time) - gid)
        grad_norms[rnn_type].append(np.linalg.norm(grad))

    dummy_loss.detach_()

plt.figure()
for rnn_type in depths:
    plt.plot(depths[rnn_type], grad_norms[rnn_type], label="%s" % rnn_type, alpha=0.7, lw=2)
plt.legend()
plt.ylabel("$ \\| \\partial \\sum_i {c_{N}}_i / \\partial c_t \\|$", fontsize=15)
plt.xlabel("Steps through time - $t$", fontsize=15)
plt.xlim((1, SEQ_LENGTH - 1))
plt.title("Gradient magnitudes across time for: RNN-Type (forget_bias value)")
# plt.savefig("mygraph.png")
plt.show()
