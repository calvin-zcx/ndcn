import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchdiffeq as ode


def row_normalization(X):
    """Row-normalize sparse matrix
       :return: D^-1 * A
    """
    X = F.normalize(X.float(), 1, 1)
    X[torch.isinf(X)] = 0

    return X


class RowNorm(nn.Module):
    def __init__(self):
        super(RowNorm, self).__init__()

    def forward(self, X):
        X = F.normalize(X.float(), 1, 1)
        X[torch.isinf(X)] = 0
        return X


class ResBlock(nn.Module):
    def __init__(self, hidden_size, A, dropout=0, normalize=False, time_varying=False, Euler=False):
        super(ResBlock, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A

        #  Other tricks
        self.normalize = normalize
        self.time_varying = time_varying
        if self.time_varying:
            self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.Euler = Euler
        if self.Euler:
            self.time_step = nn.Parameter(torch.FloatTensor([0.1]))
            nn.init.uniform_(self.time_step, 0, 1)
        else:
            self.time_step = 1

    def forward(self, x):
        shortcut = x
        if self.normalize:
            x = row_normalization(x)
        f = torch.sparse.mm(self.A, x)
        if self.time_varying:
            f = self.linear(f)
        f = self.dropout_layer(f)  # drop out for input
        if self.normalize:
            f = row_normalization(f)
        f = F.relu(f)
        return shortcut + f * self.time_step


class ODEFunc(nn.Module):
    def __init__(self, hidden_size, A, dropout=0.0):
        super(ODEFunc, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A
        self.nfe = 0
        # self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        # self.batchnorm = nn.BatchNorm1d(hidden_size)

        # self.scale = nn.Parameter(torch.FloatTensor([1]))  # [0.01]))  # np.random.rand(1) *
        # self.bias = nn.Parameter(torch.FloatTensor([1e-3]))

    def forward(self, t, x): # How to use t?
        self.nfe += 1
        f = torch.sparse.mm(self.A, x)
        # f = self.linear(f)
        f = self.dropout_layer(f)  # drop out for input
        # f = row_normalization(f)
        # f = self.batchnorm(f)
        f = F.relu(f)  # !!!!! Not use relu seems doesn't  matter!!!!!!
        return f    # self.scale + self.bias


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        # self.integration_time = torch.tensor([0, 10]).float()

        self.integration_time = torch.linspace(0, 1.9, 10).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = ode.odeint(self.odefunc, x, self.integration_time, rtol=.1, atol=.1)
        # out = ode.odeint_adjoint(self.odefunc, x, self.integration_time, rtol=1e-1, atol=1e-1)
        # return out[1]
        return out[-1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

