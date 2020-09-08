import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq as ode
from utils import *


class ODEFunc(nn.Module):  # A kind of ODECell in the view of RNN
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
        x = F.relu(x)
        # !!!!! Not use relu seems doesn't  matter!!!!!! in theory. Converge faster !!! Better than tanh??
        # x = torch.sigmoid(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False): #vt,         :param vt:
        """
        :param odefunc: X' = f(X, t, G, W)
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
        # self.integration_time_vector = vt  # time vector
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self, vt, x):
        integration_time_vector = vt.type_as(x)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # return out[-1]
        return out[-1] if self.terminal else out  # 100 * 400 * 10


class ODEBlock2(nn.Module):
    def __init__(self, odefunc, vt, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False): #vt,         :param vt:
        """
        :param odefunc: X' = f(X, t, G, W)
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

        super(ODEBlock2, self).__init__()
        self.odefunc = odefunc
        self.integration_time_vector = vt  # time vector
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self,   x):
        integration_time_vector = self.integration_time_vector.type_as(x)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # return out[-1]
        return out[-1] if self.terminal else out  # 100 * 400 * 10


class NDCN(nn.Module):  # myModel
    def __init__(self, input_size, hidden_size, A, num_classes,  dropout=0.0,
                 no_embed=False, no_graph=False, no_control=False,
                 rtol=.01, atol=.001, method='dopri5'):
        super(NDCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.A = A  # N_node * N_node
        self.num_classes = num_classes

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.no_embed = no_embed
        self.no_graph = no_graph
        self.no_control = no_control

        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.neural_dynamic_layer = ODEBlock(
                ODEFunc(hidden_size, A, dropout=dropout, no_graph=no_graph, no_control=no_control),  # OM
                rtol= rtol, atol= atol, method= method)  # t is like  continuous depth
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, vt, x):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        if not self.no_embed:
            x = self.input_layer(x)
        hvx = self.neural_dynamic_layer(vt, x)
        output = self.output_layer(hvx)
        return output


class GraphConvolution(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, input, propagation_adj):
        support = self.fc(input)
        # CAUTION: Pytorch only supports sparse * dense matrix multiplication
        # CAUTION: Pytorch does not support sparse * sparse matrix multiplication !!!
        # output = torch.sparse.mm(propagation_adj, support)
        output = torch.mm(propagation_adj, support)
        # output = torch.reshape(output, (1, -1)).contiguous()
        return output.view(1, -1)


class TemporalGCN(nn.Module):
    def __init__(self, input_size_gnn, hidden_size_gnn, input_n_graph, hidden_size_rnn, A, dropout=0.5, rnn_type='lstm'):
        super(TemporalGCN, self).__init__()
        self.input_size_gnn = input_size_gnn
        self.hidden_size_gnn = hidden_size_gnn
        self.input_size_rnn = input_n_graph * hidden_size_gnn
        self.hidden_size_rnn = hidden_size_rnn
        self.output_size = input_n_graph
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A

        self.gc = GraphConvolution(input_size_gnn, hidden_size_gnn)

        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTMCell(self.input_size_rnn, hidden_size_rnn)
        elif rnn_type =='gru':
            self.rnn = nn.GRUCell(self.input_size_rnn, hidden_size_rnn)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNNCell(self.input_size_rnn, hidden_size_rnn)

        self.linear = nn.Linear(hidden_size_rnn, input_n_graph)

    def forward(self, input, future=0):
        outputs = []
        # torch.double  h_t: 1*20  1 sample, 20 hidden in rnn
        h_t = torch.zeros(1, self.hidden_size_rnn, device=input.device)  # torch.zeros(1, self.hidden_size_rnn, dtype=torch.float)
        if self.rnn_type == 'lstm':
            c_t = torch.zeros(1, self.hidden_size_rnn, device=input.device)  # torch.zeros(1, self.hidden_size_rnn, dtype=torch.float)  # dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t = self.dropout_layer(input_t)  # input_t : 400*1
            input_t = self.gc(input_t, self.A)  # input_t 400*2 --> 1*800
            input_t = F.relu(input_t)  # 1*800
            if self.rnn_type == 'lstm':
                h_t, c_t = self.rnn(input_t, (h_t, c_t))  # input_t 1*800  h_t: 1*10
            elif self.rnn_type == 'gru':
                h_t = self.rnn(input_t, h_t )
            elif self.rnn_type == 'rnn':
                h_t = self.rnn(input_t, h_t)

            output = self.linear(h_t).t()  # 400*1
            outputs += [output]
        for i in range(future):  # if we should predict the future
            input_t = self.dropout_layer(output)  # input_t : 400*1
            input_t = self.gc(input_t, self.A)  # input_t 400*20 --> 1*8000
            input_t = F.relu(input_t)

            if self.rnn_type == 'lstm':
                h_t, c_t = self.rnn(input_t, (h_t, c_t))  # input_t 1*8000  h_t: 1*20
            elif self.rnn_type == 'gru':
                h_t = self.rnn(input_t, h_t)
            elif self.rnn_type == 'rnn':
                h_t = self.rnn(input_t, h_t)

            output = self.linear(h_t).t()
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


# class ODEBlock(nn.Module):
#     def __init__(self, odefunc, vt, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False):
#         """
#         :param odefunc: X' = f(X, t, G, W)
#         :param vt:
#         :param rtol: optional float64 Tensor specifying an upper bound on relative error,
#             per element of `y`.
#         :param atol: optional float64 Tensor specifying an upper bound on absolute error,
#             per element of `y`.
#         :param method:
#             'explicit_adams': AdamsBashforth,
#             'fixed_adams': AdamsBashforthMoulton,
#             'adams': VariableCoefficientAdamsBashforth,
#             'tsit5': Tsit5Solver,
#             'dopri5': Dopri5Solver,
#             'euler': Euler,
#             'midpoint': Midpoint,
#             'rk4': RK4,
#         """
#
#         super(ODEBlock, self).__init__()
#         self.odefunc = odefunc
#         self.integration_time_vector = vt  # time vector
#         self.rtol = rtol
#         self.atol = atol
#         self.method = method
#         self.adjoint = adjoint
#         self.terminal = terminal
#
#     def forward(self, x):
#         self.integration_time_vector = self.integration_time_vector.type_as(x)
#         if self.adjoint:
#             out = ode.odeint_adjoint(self.odefunc, x, self.integration_time_vector,
#                                      rtol=self.rtol, atol=self.atol, method=self.method)
#         else:
#             out = ode.odeint(self.odefunc, x, self.integration_time_vector,
#                              rtol=self.rtol, atol=self.atol, method=self.method)
#         # return out[-1]
#         return out[-1] if self.terminal else out  # 100 * 400 * 10

# TO BE DELETED!
# class GraphOperator(nn.Module):
#     def __init__(self,  alpha=True):
#         super(GraphOperator, self).__init__()
#         if alpha:
#             self.alpha = nn.Parameter(torch.FloatTensor([0.5]))
#         else:
#             self.register_parameter('alpha', None)
#
#     def forward(self, A, x):  # How to use t?
#         """
#         :param t:  end time tick, if t is not used, it is an autonomous system
#         :param x:  initial value   N_node * N_dim   400 * hidden_size
#         :return:
#         """
#         A_prime = self.alpha * A + (1.0-self.alpha) * torch.eye(A.shape[0]).cuda()
#         out_degree = A_prime.sum(1)
#         # in_degree = A_prime.sum(0)
#
#         out_degree_sqrt_inv = torch.matrix_power(torch.diag(out_degree), -1)
#         out_degree_sqrt_inv[torch.isinf(out_degree_sqrt_inv)] = 0.0
#         # int_degree_sqrt_inv = torch.matrix_power(torch.diag(in_degree), -0.5)
#         # int_degree_sqrt_inv[torch.isinf(int_degree_sqrt_inv)] = 0.0
#         mx_operator = torch.mm(out_degree_sqrt_inv, A_prime)
#         x = torch.mm(mx_operator, x)
#         return x
#
#
# class ODEFunc_A(nn.Module):
#     def __init__(self, hidden_size, A, dropout=0.0, no_graph=False, no_control=False):
#         super(ODEFunc_A, self).__init__()
#         self.hidden_size = hidden_size
#         self.dropout = dropout
#         self.dropout_layer = nn.Dropout(dropout)
#         self.A = A  # N_node * N_node
#         # self.nfe = 0
#         self.wt = nn.Linear(hidden_size, hidden_size)
#         self.no_graph = no_graph
#         self.no_control = no_control
#         self.GraphOperator = GraphOperator(alpha=True)
#
#     def forward(self, t, x):  # How to use t?
#         """
#         :param t:  end time tick, if t is not used, it is an autonomous system
#         :param x:  initial value   N_node * N_dim   400 * hidden_size
#         :return:
#         """
#
#         x = self.GraphOperator.forward(self.A, x)
#         if not self.no_control:
#             x = self.wt(x)
#         x = self.dropout_layer(x)
#         x = F.relu(x)
#         return x
