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

sns.set_style('ticks')


class VanillaRNNBase(nn.Module):

    def __init__(self, hidden_size, activation=nn.Tanh, bias=True):
        """
        Constructor for a simple RNNCell where the hidden-to-hidden transitions
        are defined by a linear layer and the default activation of `tanh`
        :param hidden_size: the size of the hidden state
        :param activation: the activation function used for computing the next hidden state
        """
        super(VanillaRNNBase, self).__init__()

        self._hidden_size = hidden_size
        self._activation = activation()
        self._bias = bias

        # TODO 1.1 Create the hidden-to-hidden layer
        # self._linear_hh = nn.Linear(...)
        self._linear_hh = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, inputs, hidden=None):
        out = inputs
        #### TODO 1.1 Your code here
        ### ...
        #### end code
        if not (hidden is None):
            out += self._linear_hh(hidden)
        out = self._activation(out)
        return out, out


class VanillaRNN(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=20, bias=False):
        """
        Creates a vanilla RNN where input-to-hidden is a nn.Linear layer
        and hidden-to-output is a nn.Linear layer

        :param input_size: the size of the input to the RNN
        :param hidden_size: size of the hidden state of the RNN
        :param output_size: size of the output
        """
        super(VanillaRNN, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._bias = bias

        self.in_to_hidden = nn.Linear(self._input_size, self._hidden_size, bias=self._bias)
        self.rnn_cell = VanillaRNNBase(self._hidden_size, bias=self._bias)
        self.hidden_to_out = nn.Linear(self._hidden_size, self._output_size, bias=self._bias)  # mereu acelasi bias??

    def step(self, input, hidden=None):
        ### TODO 1.2 compute one step in the RNN
        ## input_ = ....
        ## _, hidden_ =  ....
        # output_ = ....
        input_ = self.in_to_hidden(input)
        _, hidden_ = self.rnn_cell(input_, hidden)
        output_ = self.hidden_to_out(hidden_)
        return output_, hidden_
        pass

    def forward(self, inputs, hidden=None, force=True, warm_start=10):
        steps = len(inputs)

        outputs = torch.autograd.Variable(torch.zeros(steps, self._output_size, self._output_size))

        output_ = None
        hidden_ = hidden

        for i in range(steps):
            ## TODO 1.3 Implement forward pass in RNN
            ## Implement Teacher Forcing and Warm Start
            input_ = None
            input_ = inputs[i]
            ### END Code

            output_, hidden_ = self.step(input_, hidden_)
            outputs[i] = output_

        return outputs, hidden_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Running code @ {device}')

UNROLL_LENGTH = 30  # @param {type:"integer"}
NUM_ITERATIONS = 10000  # @param {type:"integer"}
WARM_START = 10  # @param {type:"integer"}
TEACHER_FORCING = False  # @param {type:"boolean"}
HIDDEN_UNITS = 20  # @param {type:"integer"}
LEARNING_RATE = 0.0001  # @param {type:"number"}
REPORTING_INTERVAL = 200  # @param {type:"integer"}

# We create training data, sine wave over [0, 2pi]
x_train = np.arange(0, 2 * np.pi, 0.1).reshape(-1, 1, 1)
y_train = np.sin(x_train)

net = VanillaRNN(hidden_size=HIDDEN_UNITS, bias=False)
net.train()
net = net.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

running_loss = 0

for iteration in range(NUM_ITERATIONS):
    # select a start point in the training set for a sequence of UNROLL_LENGTH
    start = np.random.choice(range(x_train.shape[0] - UNROLL_LENGTH))
    train_sequence = y_train[start: (start + UNROLL_LENGTH)]

    train_inputs = torch.from_numpy(train_sequence[:-1]).float().to(device)
    train_targets = torch.from_numpy(train_sequence[1:]).float().to(device)

    optimizer.zero_grad()

    outputs, hidden = net(train_inputs, hidden=None, force=TEACHER_FORCING, warm_start=WARM_START)
    outputs = outputs.to(device)
    loss = criterion(outputs, train_targets)
    loss.backward()

    running_loss += loss.item()

    optimizer.step()

    if iteration % REPORTING_INTERVAL == REPORTING_INTERVAL - 1:
        # let's see how well we do on predictions for the whole sequence
        avg_loss = running_loss / REPORTING_INTERVAL

        report_sequence = torch.from_numpy(y_train[:-1]).float().to(device)
        report_targets = torch.from_numpy(y_train[1:]).float().to(device)
        report_output, report_hidden = net(report_sequence, hidden=None, force=False, warm_start=WARM_START)
        report_output = report_output.to(device)

        report_loss = criterion(report_output, report_targets)
        print('[%d] avg_loss: %.5f, report_loss: %.5f, ' % (iteration + 1, avg_loss, report_loss.item()))

        plt.figure()
        plt.title('Training Loss %.5f;  Sampling loss %.5f; Iteration %d' % (avg_loss, report_loss.item(), iteration))

        plt.plot(y_train[1:].ravel(), c='blue', label='Ground truth',
                 linestyle=":", lw=6)
        plt.plot(range(start, start + UNROLL_LENGTH - 1), outputs.data.numpy().ravel(), c='gold',
                 label='Train prediction', lw=5, marker="o", markersize=5,
                 alpha=0.7)
        plt.plot(report_output.data.numpy().ravel(), c='r', label='Generated', lw=4, alpha=0.7)
        plt.legend()
        plt.show()