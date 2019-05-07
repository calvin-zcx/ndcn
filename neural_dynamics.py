import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq as ode


class ODEFunc(nn.Module):
    def __init__(self, hidden_size, A, dropout=0.0, no_graph=False, no_control=False):
        super(ODEFunc, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A  # N_node * N_node
        # self.nfe = 0
        self.wt = nn.Linear(hidden_size, hidden_size)
        self.no_graph = no_graph
        self.no_control = no_control

    def forward(self, t, x):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        # self.nfe += 1
        if not self.no_graph:
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                x = torch.sparse.mm(self.A, x)
            else:
                x = torch.mm(self.A, x)
        if not self.no_control:
            x = self.wt(x)
        x = self.dropout_layer(x)
        # x = torch.tanh(x)
        x = F.relu(x)  # !!!!! Not use relu seems doesn't  matter!!!!!! in theory. Converge faster !!! Better than tanh??
        # x = torch.sigmoid(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc, vt, rtol=.01, atol=.001, method='dopri5', adjoint=False):
        """
        :param odefunc: X' = f(X, t, G, W)
        :param vt:
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """

        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time_vector = vt  # time vector
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint

    def forward(self, x):
        self.integration_time_vector = self.integration_time_vector.type_as(x)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, self.integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, self.integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # return out[-1]
        return out  # 100 * 400 * 10

