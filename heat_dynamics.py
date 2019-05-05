import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import networkx as nx
import datetime
from utils_in_learn_dynamics import *
from neural_dynamics import *
import torchdiffeq as ode
import sys
import functools
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser('Heat Diffusion Dynamic Case')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--rtol', type=float, default=0.01,
                    help='optional float64 Tensor specifying an upper bound on relative error, per element of y')
parser.add_argument('--atol', type=float, default=0.001,
                    help='optional float64 Tensor specifying an upper bound on absolute error, per element of y')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=20,
                    help='Number of hidden units.')
parser.add_argument('--time_tick', type=int, default=10)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')

parser.add_argument('--n', type=int, default=400, help='Number of nodes')
parser.add_argument('--sparse', action='store_true')
parser.add_argument('--network', type=str,
                    choices=['grid', 'random', 'power_law', 'small_world', 'community'], default='grid')
parser.add_argument('--layout', type=str, choices=['community', 'degree'], default='community')
parser.add_argument('--seed', type=int, default=0, help='Random Seed')
parser.add_argument('--T', type=float, default=5., help='Terminal Time')


parser.add_argument('--baseline', type=str,
                    choices=['differential_gcn', 'no_embedding', 'no_control', 'no_graph'], default='differential_gcn')
parser.add_argument('--dump', action='store_true', help='Save Results')
parser.add_argument('--dump_appendix', type=str, default='',
                    help='dump_appendix to distinguish results file, e.g. same as baseline name')

args = parser.parse_args()
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
if args.viz:
    dirname = r'figure/heat/' + args.network
    makedirs(dirname)
    fig_title = r'Heat Diffusion Dynamics'

if args.dump:
    results_dir = r'results/heat/' + args.network
    makedirs(results_dir)

# Build network # A: Adjacency matrix, L: Laplacian Matrix,  OM: Base Operator
n = args.n # e.g nodes number 400
N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20
seed = args.seed
if args.network == 'grid':
    print("Choose graph: " + args.network)
    A = grid_8_neighbor_graph(N)
    G = nx.from_numpy_array(A.numpy())
elif args.network == 'random':
    print("Choose graph: " + args.network)
    G = nx.erdos_renyi_graph(n, 0.1, seed=seed)
    G = networkx_reorder_nodes(G, args.layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
elif args.network == 'power_law':
    print("Choose graph: " + args.network)
    G = nx.barabasi_albert_graph(n, 5, seed=seed)
    G = networkx_reorder_nodes(G,  args.layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
elif args.network == 'small_world':
    print("Choose graph: " + args.network)
    G = nx.newman_watts_strogatz_graph(400, 5, 0.5, seed=seed)
    G = networkx_reorder_nodes(G, args.layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))
elif args.network == 'community':
    print("Choose graph: " + args.network)
    n1 = int(n/3)
    n2 = int(n/3)
    n3 = int(n/4)
    n4 = n - n1 - n2 -n3
    G = nx.random_partition_graph([n1, n2, n3, n4], .25, .01, seed=seed)
    G = networkx_reorder_nodes(G, args.layout)
    A = torch.FloatTensor(nx.to_numpy_array(G))

if args.viz:
    visualize_graph_matrix(G, args.network)

D = torch.diag(A.sum(1))
L = (D - A)
t = torch.linspace(0., args.T, args.time_tick)  # args.time_tick) # 100 vector
# OM = torch.FloatTensor(zipf_smoothing(A.numpy()))
OM = torch.FloatTensor(normalized_laplacian(A.numpy()))

if args.sparse:
    # For small network, dense matrix is faster
    # For large network, sparse matrix cause less memory
    L = torch_sensor_to_torch_sparse_tensor(L)
    A = torch_sensor_to_torch_sparse_tensor(A)
    OM = torch_sensor_to_torch_sparse_tensor(OM)

# Initial Value
x0 = torch.zeros(N, N)
x0[int(0.05*N):int(0.25*N), int(0.05*N):int(0.25*N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
x0[int(0.45*N):int(0.75*N), int(0.45*N):int(0.75*N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
x0[int(0.05*N):int(0.25*N), int(0.35*N):int(0.65*N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case
x0 = x0.view(-1, 1).float()
energy = x0.sum()


class HeatDiffusion(nn.Module):
    # In this code, row vector: y'^T = y^T A^T      textbook format: column vector y' = A y
    def __init__(self,  L,  k=1):
        super(HeatDiffusion, self).__init__()
        self.L = -L  # Diffusion operator
        self.k = k   # heat capacity

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dX(t)/dt = -k * L *X
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if hasattr(self.L, 'is_sparse') and self.L.is_sparse:
            f = torch.sparse.mm(self.L, x)
        else:
            f = torch.mm(self.L, x)
        return self.k * f


with torch.no_grad():
    solution_numerical = ode.odeint(HeatDiffusion(L, 1), x0, t, method='dopri5')  # shape: 1000 * 1 * 2
    print(solution_numerical.shape)


now = datetime.datetime.now()
appendix = now.strftime("%m%d-%H%M%S")
zmin = solution_numerical.min()
zmax = solution_numerical.max()
for ii, xt in enumerate(solution_numerical, start=1):
    if args.viz:
        print(xt.shape)
        visualize(N, x0, xt, '{:03d}-tru'.format(ii)+appendix, fig_title, dirname, zmin, zmax)


true_y = solution_numerical.squeeze().t().to(device)  # 100 * 1 * 400  --squeeze--> 100 * 400 -t-> 400 * 100
true_y0 = x0.to(device)  # 400 * 1
L = L.to(device)  # 400 * 400
OM = OM.to(device)  # 400 * 400
A = A.to(device)

# Build model
input_size = true_y0.shape[1]   # y0: 400*1 ,  input_size:1
hidden_size = args.hidden  # args.hidden  # 20 default  # [400 * 1 ] * [1 * 20] = 400 * 20
dropout = args.dropout  # 0 default, not stochastic ODE
num_classes = 1  # 1 for regression

# choices=['differential_gcn', 'no_embedding', 'no_control', 'no_graph']
if args.baseline == 'differential_gcn':
    print('Choose model:' + args.baseline)
    embedding_layer = [nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),  #nn.ReLU(inplace=True), #
                        nn.Linear(hidden_size, hidden_size, bias=True)]
    neural_dynamic_layer = [ODEBlock(
        ODEFunc(hidden_size, OM, dropout=dropout),  # OM
        t,
        rtol=args.rtol, atol=args.atol, method=args.method)]  # t is like  continuous depth
    semantic_layer = [nn.Linear(hidden_size, num_classes, bias=True)]

elif args.baseline == 'no_embedding':
    print('Choose model:' + args.baseline)
    embedding_layer = []
    neural_dynamic_layer = [ODEBlock(
        ODEFunc(input_size, OM, dropout=dropout),
        t,
        rtol=args.rtol, atol=args.atol, method=args.method)]  # t is like  continuous depth
    semantic_layer = [nn.Linear(input_size, num_classes, bias=True)]

elif args.baseline == 'no_control':
    print('Choose model:' + args.baseline)
    embedding_layer = [nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),  # nn.ReLU(inplace=True),  #
                       nn.Linear(hidden_size, hidden_size, bias=True)]
    neural_dynamic_layer = [ODEBlock(
        ODEFunc(hidden_size, OM, dropout=dropout, no_control=True),
        t,
        rtol=args.rtol, atol=args.atol, method=args.method)]  # t is like  continuous depth
    semantic_layer = [nn.Linear(hidden_size, num_classes, bias=True)]

elif args.baseline == 'no_graph':
    print('Choose model:' + args.baseline)
    embedding_layer = [nn.Linear(input_size, hidden_size, bias=True),  nn.Tanh(),  # nn.ReLU(inplace=True), #
                       nn.Linear(hidden_size, hidden_size, bias=True)]
    neural_dynamic_layer = [ODEBlock(
        ODEFunc(hidden_size, OM, dropout=dropout, no_graph=True),
        t,
        rtol=args.rtol, atol=args.atol, method=args.method)]  # t is like  continuous depth
    semantic_layer = [nn.Linear(hidden_size, num_classes, bias=True)]

model = nn.Sequential(*embedding_layer, *neural_dynamic_layer, *semantic_layer).to(device)


if __name__ == '__main__':
    t_start = time.time()
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = F.l1_loss  # F.mse_loss(pred_y, true_y)
    # time_meter = RunningAverageMeter(0.97)
    # loss_meter = RunningAverageMeter(0.97)
    if args.dump:
        results_dict = {
            'args': args.__dict__,
            'v_iter': [],
            'abs_error': [],
            'rel_error': [],
            'true_y': [solution_numerical.squeeze().t()],
            'predict_y': [],
            'model_state_dict': [],
            'total_time': []}

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_y = model(true_y0)  # 20 * 400 * 1 should be 400 * 20
        pred_y = pred_y.squeeze().t()
        loss = criterion(pred_y, true_y)  # 400 * 20 (time_tick)   # torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        # time_meter.update(time.time() - t_start)
        # loss_meter.update(loss.item())
        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = model(true_y0).squeeze().t() # odeint(model, true_y0, t)
                loss = criterion(pred_y, true_y)
                relative_loss = criterion(pred_y, true_y) / true_y.mean()
                if args.dump:
                    # Info to dump
                    results_dict['v_iter'].append(itr)
                    results_dict['abs_error'].append(loss.item())    # {'abs_error': [], 'rel_error': [], 'X_t': []}
                    results_dict['rel_error'].append(relative_loss.item())
                    results_dict['predict_y'].append(pred_y)
                    results_dict['model_state_dict'].append(model.state_dict())
                    # now = datetime.datetime.now()
                    # appendix = now.strftime("%m%d-%H%M%S")
                    # results_dict_path = results_dir + r'/result_' + appendix + '.' + args.dump_appendix
                    # torch.save(results_dict, results_dict_path)
                    # print('Dump results as: ' + results_dict_path)

                print('Iter {:04d} | Total Loss {:.6f} | Relative Loss {:.6f} | Time {:.4f}'
                      .format(itr, loss.item(), relative_loss.item(), time.time() - t_start))

    now = datetime.datetime.now()
    appendix = now.strftime("%m%d-%H%M%S")
    with torch.no_grad():
        pred_y = model(true_y0).squeeze().t()  # odeint(model, true_y0, t)
        loss = criterion(pred_y, true_y)
        relative_loss = criterion(pred_y, true_y) / true_y.mean()
        print('Iter {:04d} | Total Loss {:.6f} | Relative Loss {:.6f} | Time {:.4f}'
              .format(itr, loss.item(), relative_loss.item(), time.time() - t_start))

        if args.viz:
            for ii in range(pred_y.shape[1]):
                xt_pred = pred_y[:, ii].cpu()
                # print(xt_pred.shape)
                visualize(N, x0, xt_pred,
                          '{:03d}-{:s}-'.format(ii+1, args.dump_appendix)+appendix,
                          fig_title, dirname, zmin, zmax)

        t_total = time.time() - t_start
        print('Total Time {:.4f}'.format(t_total))
        if args.dump:
            results_dict['total_time'] = t_total
            results_dict_path = results_dir + r'/result_' + appendix + '.' + args.dump_appendix
            torch.save(results_dict, results_dict_path)
            print('Dump results as: ' + results_dict_path)

    # Test dumped results:
    rr = torch.load(results_dict_path)
    fig, ax = plt.subplots()
    ax.plot(rr['v_iter'], rr['abs_error'], '-', label='Absolute Error')
    ax.plot(rr['v_iter'], rr['rel_error'], '--', label='Relative Error')
    legend = ax.legend( fontsize='x-large') # loc='upper right', shadow=True,
    # legend.get_frame().set_facecolor('C0')
    if args.dump:
        fig.savefig(results_dict_path + ".png", transparent=True)
        fig.savefig(results_dict_path + ".pdf", transparent=True)
    plt.show()
    plt.pause(0.001)
    plt.close(fig)

# --time_tick 20 --niters 2500 --network grid --dump --dump_appendix differential_gcn --baseline differential_gcn  --viz
# python heat_dynamics.py  --time_tick 20 --niters 2500 --network grid --dump --dump_appendix differential_gcn --baseline differential_gcn  --viz
