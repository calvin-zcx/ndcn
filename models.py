import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GraphConvolution(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, input, propagation_adj):
        support = self.fc(input)
        # CAUTION: Pytorch only supports sparse * dense matrix multiplication
        # CAUTION: Pytorch does not support sparse * sparse matrix multiplication !!!
        output = torch.sparse.mm(propagation_adj, support)

        return output


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0, num_middle_layers=0):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input_size, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, num_classes)
        self.dropout = dropout

        self.conv_middle= nn.ModuleList([GraphConvolution(hidden_size, hidden_size) for i in range(num_middle_layers)])

    def forward(self, x, propagation_adj):

        x = F.dropout(x, self.dropout, training=self.training)  # drop out for input
        x = self.gc1.forward(x, propagation_adj)
        x = F.relu(x)

        for conv_middle in self.conv_middle:
            x = F.dropout(x, self.dropout, training=self.training)  # drop out for input
            x = conv_middle.forward(x, propagation_adj)
            x = F.relu(x)

        x = F.dropout(x, self.dropout, training=self.training)  # drop out for hidden layers
        x = self.gc2.forward(x, propagation_adj)

        return x


class DeepGCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0, num_middle_layers=0):
        super(DeepGCN, self).__init__()

        self.conv1 = GraphConvolution(input_size, hidden_size)
        self.conv_middle = nn.ModuleList([GraphConvolution(hidden_size, hidden_size) for i in range(num_middle_layers)])
        self.conv2 = GraphConvolution(hidden_size, num_classes)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        self.gc1Linear = nn.Linear(input_size, hidden_size, bias=False)
        self.gc2Linear = nn.Linear(hidden_size, num_classes, bias=False)
        self.time_step = nn.Parameter(torch.FloatTensor([0.1]))   # [0.01]))  # np.random.rand(1) *

        # self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        # self.bn2 = nn.BatchNorm1d(num_features=num_classes)

    def forward(self, x, propagation_adj):
        """
        :param x:
        :param propagation_adj:
        :return: x(t+\delta) = x(t) + f(x(t)) * \delta
        """

        # x = F.dropout(x, self.dropout, training=self.training)  # drop out for input
        # x = self.gc1.forward(x, propagation_adj)
        # residual = self.gc1Linear.forward(x)

        # x = F.dropout(x, self.dropout, training=self.training)  # drop out for input
        # x = F.dropout(x, 0.2, training=self.training)  # drop out for input
        x = self.dropout_layer(x)
        x = self.conv1.forward(x, propagation_adj)
        # x = self.bn1(x)
        x = F.relu(x)
        # x += residual
        # x = x * self.time_step + residual

        for conv_middle in self.conv_middle:
            f = F.dropout(x, self.dropout, training=self.training)  # drop out for input
            f = conv_middle.forward(f, propagation_adj)
            f = F.relu(f)
            x = x + f * self.time_step

        # x = self.gc2Linear.forward(x)
        # f = self.gc2Linear.forward(x)
        x = self.dropout_layer(x)
        # x = F.dropout(x, self.dropout, training=self.training)  # drop out for hidden layers
        x = self.conv2.forward(x, propagation_adj)
        # x = F.relu(x)
        # x = self.bn2(x)

        # x = self.gc2Linear.forward(x)
        # x += f
        # x = x + f * self.time_step

        return x


class DeepGCN2(nn.Module):
    def __init__(self, filter_matrix,  input_size, hidden_size, num_classes, dropout=0, num_middle_layers=0):
        super(DeepGCN2, self).__init__()
        self.filter_matrix = filter_matrix
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        # self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, num_classes, bias=True)
        self.time_step = nn.Parameter(torch.FloatTensor([0.1]))  # [0.01]))  # np.random.rand(1) *

    def forward(self, x, propagation_adj):
        """
        :param x:
        :param propagation_adj:
        :return: x(t+\delta) = x(t) + f(x(t)) * \delta
        """
        x = torch.sparse.mm(self.filter_matrix, x)
        x = self.dropout_layer(x)
        x = self.linear1(x)
        x = F.relu(x)

        # for conv_middle in self.conv_middle:
        #     f = F.dropout(x, self.dropout, training=self.training)  # drop out for input
        #     f = conv_middle.forward(f, propagation_adj)
        #     f = F.relu(f)
        #     x = x + f * self.time_step

        x = torch.sparse.mm(self.filter_matrix, x)
        x = self.dropout_layer(x)
        x = self.linear2(x)
        # x = F.relu(x)

        return x


class DeepGCN3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_nodes, dropout=0, num_middle_layers=0):
        super(DeepGCN3, self).__init__()
        # self.filter_matrix = filter_matrix
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        # self.dropout = dropout
        self.conv_middle = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for i in range(num_middle_layers)])
        self.dropout_layer = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, num_classes, bias=True)
        self.linear_oldDim = nn.Linear(input_size, num_classes, bias=True)
        self.time_step = nn.Parameter(torch.FloatTensor([0.1]))  # [0.01]))  # np.random.rand(1) *

        #self.A = adj.to_dense()
        self.AW = nn.Parameter(torch.FloatTensor(torch.rand(num_nodes, num_nodes)))

        # indices = torch.from_numpy(
        # np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        # values = torch.from_numpy(sparse_mx.data)
        # shape = torch.Size(sparse_mx.shape)
        # torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, x, propagation_adj):
        """
        :param x:
        :param propagation_adj:
        :return: x(t+\delta) = x(t) + f(x(t)) * \delta
        """
        # x = torch.sparse.mm(self.filter_matrix, x)
        # x = self.dropout_layer(x)
        x = self.linear1(x)
        # x = F.relu(x)

        A = self.AW * propagation_adj
        D = torch.diag(A.sum(1))
        L = A - D

        for conv_middle in self.conv_middle:
            # f = torch.sparse.mm(propagation_adj, x)
            f = torch.mm(L, x)
            # f = self.dropout_layer(f)  # drop out for input
            # f = conv_middle.forward(f)
            f = F.relu(f)
            # f = conv_middle.forward(f)
            # f = torch.sparse.mm(propagation_adj, f)
            # f = self.dropout_layer(f)  # drop out for input
            # f = F.relu(f)
            x = x + f * self.time_step
            # x = conv_middle.forward(x)
            # x = F.relu(x)
            # x = conv_middle.forward(x)
            # x = F.relu(x)

        # x = torch.sparse.mm(self.filter_matrix, x)
        # x = self.dropout_layer(x)
        x = self.linear2(x)
        # x = self.linear_oldDim(x)
        # x = F.relu(x)

        return x


class DiagLinear(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, n_features, bias=True):
        super(DiagLinear, self).__init__()
        self.n_features = n_features
        self.weight = nn.Parameter(torch.FloatTensor(n_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(n_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # self.weight = nn.Parameter(torch.diag(self.weight))
        # print(self.weight.shape)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.n_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.spmm(input, torch.diag(self.weight))
        # output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class DeepGCN4(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0, num_middle_layers=0):
        super(DeepGCN4, self).__init__()
        # self.filter_matrix = filter_matrix
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        # self.dropout = dropout
        self.conv_middle = nn.ModuleList(
            [DiagLinear(hidden_size,  bias=False) for i in range(num_middle_layers)])
        # self.conv_middle = nn.ModuleList(
        #     [nn.Linear(hidden_size, hidden_size, bias=True) for i in range(num_middle_layers)])
        self.dropout_layer = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, num_classes, bias=True)
        self.linear_oldDim = nn.Linear(input_size, num_classes, bias=True)
        self.time_step = nn.Parameter(torch.FloatTensor([0.1]))  # [0.01]))  # np.random.rand(1) *
        self.time_step_list = nn.Parameter(torch.FloatTensor([0.1 for i in range(num_middle_layers)]))  # [0.01]))  # np.random.rand(1) *

    def forward(self, x, propagation_adj):
        """
        :param x:
        :param propagation_adj:
        :return: x(t+\delta) = x(t) + f(x(t)) * \delta
        """
        # x = torch.sparse.mm(propagation_adj, x)
        # x = self.dropout_layer(x)
        x = self.linear1(x)
        x = F.relu(x)
        # x = torch.sigmoid(x)

        nl = 0
        for conv_middle in self.conv_middle:
            f = torch.sparse.mm(propagation_adj, x)
            f = self.dropout_layer(f)  # drop out for input
            # f = conv_middle.forward(f)
            f = F.relu(f)
            # f = F.tanh(f)
            # f = torch.sigmoid(f)
            # f = conv_middle.forward(f)
            # f = torch.sparse.mm(propagation_adj, f)
            # f = self.dropout_layer(f)  # drop out for input
            # f = F.relu(f)
            # x = x + f * self.time_step
            x = x + f * self.time_step_list[nl]
            # x = f
            nl += 1

        # x = torch.sparse.mm(self.filter_matrix, x)
        # x = self.dropout_layer(x)
        x = self.linear2(x)
        # x = self.linear_oldDim(x)
        # x = F.relu(x)

        return x



