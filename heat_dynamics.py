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
parser.add_argument('--method', type=str,
                    choices=['dopri5', 'adams', 'explicit_adams', 'fixed_adams','tsit5', 'euler', 'midpoint', 'rk4'],
                    default='euler')  # dopri5
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
parser.add_argument('--time_tick', type=int, default=100) # default=10)
parser.add_argument('--sampled_time', type=str,
                    choices=['irregular', 'equal'], default='irregular')

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
parser.add_argument('--operator', type=str,
                    choices=['lap', 'norm_lap', 'kipf', 'norm_adj' ], default='norm_lap')

parser.add_argument('--baseline', type=str,
                    choices=['ndcn', 'no_embed', 'no_control', 'no_graph',
                             'lstm_gnn', 'rnn_gnn', 'gru_gnn'],
                    default='differential_gcn')
parser.add_argument('--dump', action='store_true', help='Save Results')
# parser.add_argument('--dump_appendix', type=str, default='',
#                    help='dump_appendix to distinguish results file, e.g. same as baseline name')
# use args.baseline instead

args = parser.parse_args()
if args.gpu >= 0:
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

if args.viz:
    dirname = r'figure/heat/' + args.network
    makedirs(dirname)
    fig_title = r'Heat Diffusion Dynamics'

if args.dump:
    results_dir = r'results/heat/' + args.network
    makedirs(results_dir)

# Build network # A: Adjacency matrix, L: Laplacian Matrix,  OM: Base Operator
n = args.n  # e.g nodes number 400
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
    makedirs(r'figure/network/')
    visualize_graph_matrix(G, args.network)

D = torch.diag(A.sum(1))
L = (D - A)

# equally-sampled time
# sampled_time = 'irregular'
if args.sampled_time == 'equal':
    print('Build Equally-sampled -time dynamics')
    t = torch.linspace(0., args.T, args.time_tick)  # args.time_tick) # 100 vector
    # train_deli = 80
    id_train = list(range(int(args.time_tick * 0.8))) # first 80 % for train
    id_test = list(range(int(args.time_tick * 0.8), args.time_tick)) # last 20 % for test (extrapolation)
    t_train = t[id_train]
    t_test = t[id_test]
elif args.sampled_time == 'irregular':
    print('Build irregularly-sampled -time dynamics')
    # irregular time sequence
    sparse_scale = 10
    t = torch.linspace(0., args.T, args.time_tick * sparse_scale) # 100 * 10 = 1000 equally-sampled tick
    t = np.random.permutation(t)[:int(args.time_tick * 1.2)]
    t = torch.tensor(np.sort(t))
    t[0] = 0
    # t is a 120 dim irregularly-sampled time stamps

    id_test = list(range(args.time_tick, int(args.time_tick * 1.2)))  # last 20 beyond 100 for test (extrapolation)
    id_test2 = np.random.permutation(range(1, args.time_tick))[:int(args.time_tick * 0.2)].tolist()
    id_test2.sort() # first 20  in 100 for interpolation
    id_train = list(set(range(args.time_tick)) - set(id_test2))  # first 80 in 100 for train
    id_train.sort()

    t_train = t[id_train]
    t_test = t[id_test]
    t_test2 = t[id_test2]


if args.operator == 'lap':
    print('Graph Operator: Laplacian')
    OM = L
elif args.operator == 'kipf':
    print('Graph Operator: Kipf')
    OM = torch.FloatTensor(zipf_smoothing(A.numpy()))
elif args.operator == 'norm_adj':
    print('Graph Operator: Normalized Adjacency')
    OM = torch.FloatTensor(normalized_adj(A.numpy()))
else:
    print('Graph Operator[Default]: Normalized Laplacian')
    OM = torch.FloatTensor(normalized_laplacian(A.numpy()))  # L # normalized_adj


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


true_y = solution_numerical.squeeze().t().to(device)  # 120 * 1 * 400  --squeeze--> 120 * 400 -t-> 400 * 120
true_y0 = x0.to(device)  # 400 * 1
true_y_train = true_y[:, id_train] # 400*80  for train
true_y_test = true_y[:, id_test] # 400*20  for extrapolation prediction
if args.sampled_time == 'irregular':
    true_y_test2 = true_y[:, id_test2]  # 400*20  for interpolation prediction
L = L.to(device)  # 400 * 400
OM = OM.to(device)  # 400 * 400
A = A.to(device)

# Build model
input_size = true_y0.shape[1]   # y0: 400*1 ,  input_size:1
hidden_size = args.hidden  # args.hidden  # 20 default  # [400 * 1 ] * [1 * 20] = 400 * 20
dropout = args.dropout  # 0 default, not stochastic ODE
num_classes = 1  # 1 for regression
# Params for discrete models
input_n_graph= true_y0.shape[0]
hidden_size_gnn = 5
hidden_size_rnn = 20


flag_model_type = ""  # "continuous" "discrete"  input, model, output format are little different
# Continuous time network dynamic models
if args.baseline == 'ndcn':
    print('Choose model:' + args.baseline)
    flag_model_type = "continuous"
    model = NDCN(input_size=input_size, hidden_size=hidden_size, A=OM, num_classes=num_classes,
                 dropout=dropout, no_embed=False, no_graph=False, no_control=False,
                 rtol=args.rtol, atol=args.atol, method=args.method)
elif args.baseline == 'no_embed':
    print('Choose model:' + args.baseline)
    flag_model_type = "continuous"
    model = NDCN(input_size=input_size, hidden_size=input_size, A=OM, num_classes=num_classes,
                 dropout=dropout, no_embed=True, no_graph=False, no_control=False,
                 rtol=args.rtol, atol=args.atol, method=args.method)
elif args.baseline == 'no_control':
    print('Choose model:' + args.baseline)
    flag_model_type = "continuous"
    model = NDCN(input_size=input_size, hidden_size=hidden_size, A=OM, num_classes=num_classes,
                 dropout=dropout, no_embed=False, no_graph=False, no_control=True,
                 rtol=args.rtol, atol=args.atol, method=args.method)
elif args.baseline == 'no_graph':
    print('Choose model:' + args.baseline)
    flag_model_type = "continuous"
    model = NDCN(input_size=input_size, hidden_size=hidden_size, A=OM, num_classes=num_classes,
                 dropout=dropout, no_embed=False, no_graph=True, no_control=False,
                 rtol=args.rtol, atol=args.atol, method=args.method)
# Discrete time or Sequential network dynamic models
elif args.baseline == 'lstm_gnn':
    print('Choose model:' + args.baseline)
    flag_model_type = "discrete"
    print('Graph Operator: Kipf') # Using GCN as graph embedding layer
    OM = torch.FloatTensor(zipf_smoothing(A.numpy()))
    OM = OM.to(device)
    model = TemporalGCN(input_size, hidden_size_gnn, input_n_graph, hidden_size_rnn, OM, dropout=dropout, rnn_type='lstm')
elif args.baseline == 'gru_gnn':
    print('Choose model:' + args.baseline)
    flag_model_type = "discrete"
    print('Graph Operator: Kipf')
    OM = torch.FloatTensor(zipf_smoothing(A.numpy()))
    OM = OM.to(device)
    model = TemporalGCN(input_size, hidden_size_gnn, input_n_graph, hidden_size_rnn, OM, dropout=dropout, rnn_type='gru')
elif args.baseline == 'rnn_gnn':
    print('Choose model:' + args.baseline)
    flag_model_type = "discrete"
    print('Graph Operator: Kipf')
    OM = torch.FloatTensor(zipf_smoothing(A.numpy()))
    OM = OM.to(device)
    model = TemporalGCN(input_size, hidden_size_gnn, input_n_graph, hidden_size_rnn, OM, dropout=dropout, rnn_type='rnn')


# model = nn.Sequential(*embedding_layer, *neural_dynamic_layer, *semantic_layer).to(device)

num_paras = get_parameter_number(model)

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
            'abs_error2': [],
            'rel_error2': [],
            'predict_y2': [],
            'model_state_dict': [],
            'total_time': []}

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()

        if flag_model_type == "continuous":
            pred_y = model(t_train, true_y0)  # 80 * 400 * 1 should be 400 * 80
            pred_y = pred_y.squeeze().t()
            loss_train = criterion(pred_y, true_y_train) # true_y)  # 400 * 20 (time_tick)
            # torch.mean(torch.abs(pred_y - batch_y))
            relative_loss_train = criterion(pred_y, true_y_train) / true_y_train.mean()
        elif flag_model_type == "discrete":
            # true_y_train = true_y[:, id_train]  # 400*80  for train
            pred_y = model(true_y_train[:, :-1])  # true_y_train 400*80 true_y_train[:, :-1] 400*79
            # pred_y = pred_y.squeeze().t()
            loss_train = criterion(pred_y, true_y_train[:, 1:])  # true_y)  # 400 * 20 (time_tick)
            # torch.mean(torch.abs(pred_y - batch_y))
            relative_loss_train = criterion(pred_y, true_y_train[:, 1:]) / true_y_train[:, 1:].mean()
        else:
            print("flag_model_type NOT DEFINED!")
            exit(-1)

        loss_train.backward()
        optimizer.step()

        # time_meter.update(time.time() - t_start)
        # loss_meter.update(loss.item())
        if itr % args.test_freq == 0:
            with torch.no_grad():
                if flag_model_type == "continuous":
                    # pred_y = model(true_y0).squeeze().t() # odeint(model, true_y0, t)
                    # loss = criterion(pred_y, true_y)
                    # relative_loss = criterion(pred_y, true_y) / true_y.mean()
                    pred_y = model(t, true_y0).squeeze().t()  # odeint(model, true_y0, t)
                    loss = criterion(pred_y[:, id_test], true_y_test)
                    relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()
                    if args.sampled_time == 'irregular': # for interpolation results
                        loss2 = criterion(pred_y[:, id_test2], true_y_test2)
                        relative_loss2 = criterion(pred_y[:, id_test2], true_y_test2) / true_y_test2.mean()
                elif flag_model_type == "discrete":
                    pred_y = model(true_y_train, future=len(id_test)) #400*100
                    # pred_y = pred_y.squeeze().t()
                    loss = criterion(pred_y[:, id_test], true_y_test) #pred_y[:, id_test] 400*20
                    # torch.mean(torch.abs(pred_y - batch_y))
                    relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()

                if args.dump:
                    # Info to dump
                    results_dict['v_iter'].append(itr)
                    results_dict['abs_error'].append(loss.item())    # {'abs_error': [], 'rel_error': [], 'X_t': []}
                    results_dict['rel_error'].append(relative_loss.item())
                    results_dict['predict_y'].append(pred_y[:, id_test])
                    results_dict['model_state_dict'].append(model.state_dict())
                    if args.sampled_time == 'irregular':  # for interpolation results
                        results_dict['abs_error2'].append(loss2.item())  # {'abs_error': [], 'rel_error': [], 'X_t': []}
                        results_dict['rel_error2'].append(relative_loss2.item())
                        results_dict['predict_y2'].append(pred_y[:, id_test2])
                    # now = datetime.datetime.now()
                    # appendix = now.strftime("%m%d-%H%M%S")
                    # results_dict_path = results_dir + r'/result_' + appendix + '.' + args.dump_appendix
                    # torch.save(results_dict, results_dict_path)
                    # print('Dump results as: ' + results_dict_path)
                if args.sampled_time == 'irregular':
                    print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                          '| Test Loss {:.6f}({:.6f} Relative) '
                          '| Test Loss2 {:.6f}({:.6f} Relative) '
                          '| Time {:.4f}'
                          .format(itr, loss_train.item(), relative_loss_train.item(),
                                  loss.item(), relative_loss.item(),
                                  loss2.item(), relative_loss2.item(),
                                  time.time() - t_start))
                else:
                    print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                          '| Test Loss {:.6f}({:.6f} Relative) '
                          '| Time {:.4f}'
                          .format(itr, loss_train.item(), relative_loss_train.item(),
                                  loss.item(), relative_loss.item(),
                                  time.time() - t_start))

    now = datetime.datetime.now()
    appendix = now.strftime("%m%d-%H%M%S")
    with torch.no_grad():
        if flag_model_type == "continuous":
            pred_y = model(t, true_y0).squeeze().t()  # odeint(model, true_y0, t)
            loss = criterion(pred_y[:, id_test], true_y_test)
            relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()
            if args.sampled_time == 'irregular':  # for interpolation results
                loss2 = criterion(pred_y[:, id_test2], true_y_test2)
                relative_loss2 = criterion(pred_y[:, id_test2], true_y_test2) / true_y_test2.mean()
        elif flag_model_type == "discrete":
            pred_y = model(true_y_train, future=len(id_test))  # 400*100
            loss = criterion(pred_y[:, id_test], true_y_test)  # pred_y[:, id_test] 400*20
            relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()

        if args.sampled_time == 'irregular':
            print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                  '| Test Loss {:.6f}({:.6f} Relative) '
                  '| Test Loss2 {:.6f}({:.6f} Relative) '
                  '| Time {:.4f}'
                  .format(itr, loss_train.item(), relative_loss_train.item(),
                          loss.item(), relative_loss.item(),
                          loss2.item(), relative_loss2.item(),
                          time.time() - t_start))
        else:
            print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                  '| Test Loss {:.6f}({:.6f} Relative) '
                  '| Time {:.4f}'
                  .format(itr, loss_train.item(), relative_loss_train.item(),
                          loss.item(), relative_loss.item(),
                          time.time() - t_start))

        if args.viz:
            for ii in range(pred_y.shape[1]):
                xt_pred = pred_y[:, ii].cpu()
                # print(xt_pred.shape)
                visualize(N, x0, xt_pred,
                          '{:03d}-{:s}-'.format(ii+1, args.dump_appendix)+appendix,
                          fig_title, dirname, zmin, zmax)

        t_total = time.time() - t_start
        print('Total Time {:.4f}'.format(t_total))
        num_paras = get_parameter_number(model)
        if args.dump:
            results_dict['total_time'] = t_total
            results_dict_path = results_dir + r'/result_' + appendix + '.' + args.baseline  #args.dump_appendix
            torch.save(results_dict, results_dict_path)
            print('Dump results as: ' + results_dict_path)

            # Test dumped results:
            rr = torch.load(results_dict_path)
            fig, ax = plt.subplots()
            ax.plot(rr['v_iter'], rr['abs_error'], '-', label='Absolute Error')
            ax.plot(rr['v_iter'], rr['rel_error'], '--', label='Relative Error')
            legend = ax.legend( fontsize='x-large') # loc='upper right', shadow=True,
            # legend.get_frame().set_facecolor('C0')
            fig.savefig(results_dict_path + ".png", transparent=True)
            fig.savefig(results_dict_path + ".pdf", transparent=True)
            plt.show()
            plt.pause(0.001)
            plt.close(fig)

# --time_tick 20 --niters 2500 --network grid --dump --dump_appendix differential_gcn --baseline differential_gcn  --viz
# python heat_dynamics.py  --time_tick 20 --niters 2500 --network grid --dump --dump_appendix differential_gcn --baseline differential_gcn  --viz
