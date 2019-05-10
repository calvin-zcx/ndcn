import argparse
import time
import datetime
# import torch
import torch.nn.functional as F
import torch.optim as optim

from models import *
from ode_gcn import *

import pandas as pd

# import propagation as prp
# import scipy.sparse as sp
# import numpy as np

from utils import *
from sms import *
from neural_dynamics import *

import torchdiffeq as ode
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=-1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--rtol', type=float, default=0.1, #0.01,
                    help='optional float64 Tensor specifying an upper bound on relative error, per element of y')
parser.add_argument('--atol', type=float, default=0.1, #0.001,
                    help='optional float64 Tensor specifying an upper bound on absolute error, per element of y')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('-nhl', '--nHiddenLayers', type=int, default=0, help='Number of Hidden layers.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora",
                    help='Dataset to use.')
parser.add_argument('--model', type=str, default="DeepGCN",
                    choices=["DeepGCN", "GCN", "DeepGCN2", "DeepGCN3", "DeepGCN4", "resGCN", "odeGCN", "differential_gcn"],
                    help='model to use.')
parser.add_argument('--iter', type=int, default=1, help='Number of experiments to conduct')
parser.add_argument('--dump', action='store_true', default=False,
                    help='Dump results to time appendix file.')
parser.add_argument('--delta', type=float, default=1.0, help='Scale of signals from neighborhoods')

parser.add_argument('--sms', action='store_true', default=False,
                    help='Send results short message to my Phone.')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='Row normalize the feature in residual block')
parser.add_argument('--Euler', action='store_true', default=False,
                    help='Euler step in forward method')
parser.add_argument('--T', type=float, default=2., help='Terminal Time')
parser.add_argument('--time_tick', type=int, default=5)
parser.add_argument('--no_control', action='store_true', help='No control in DYnamics')

args, _ = parser.parse_known_args()
# Test if we can use GPU
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set random seed for debug and reproduce
if args.seed != -1:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

T_VERY_BEGINING = time.time()
# Input dataset
adj, features, labels, idx_train, idx_val, idx_test = load_data("cora", args.delta)

if args.cuda:
    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


# Model and optimizer
if args.model == "GCN":
    model = GCN(input_size=features.shape[1],
                hidden_size=args.hidden,
                num_classes=labels.max().item() + 1,
                dropout=args.dropout,
                num_middle_layers=args.nHiddenLayers)
elif args.model == "DeepGCN":
    model = DeepGCN(input_size=features.shape[1],
                    hidden_size=args.hidden,
                    num_classes=labels.max().item() + 1,
                    dropout=args.dropout,
                    num_middle_layers=args.nHiddenLayers)
elif args.model == 'DeepGCN2':
    model = DeepGCN2(adj,
                     input_size=features.shape[1],
                     hidden_size=args.hidden,
                     num_classes=labels.max().item() + 1,
                     dropout=args.dropout,
                     num_middle_layers=args.nHiddenLayers)
elif args.model == 'DeepGCN3':
    model = DeepGCN3(input_size=features.shape[1],
                     hidden_size=args.hidden,
                     num_classes=labels.max().item() + 1,
                     num_nodes= features.shape[0],
                     dropout=args.dropout,
                     num_middle_layers=args.nHiddenLayers)
    adj = adj.to_dense()
elif args.model == 'DeepGCN4':
    model = DeepGCN4(input_size=features.shape[1],
                     hidden_size=args.hidden,
                     num_classes=labels.max().item() + 1,
                     dropout=args.dropout,
                     num_middle_layers=args.nHiddenLayers)
elif args.model == 'resGCN':
    input_size = features.shape[1]
    hidden_size = args.hidden
    num_classes = labels.max().item() + 1
    dropout = args.dropout
    normalize = args.normalize
    nhl = args.nHiddenLayers
    Euler = args.Euler
    in_layer = [nn.Linear(input_size, hidden_size, bias=True), nn.ReLU(inplace=True)]
    feature_layer = [ResBlock(hidden_size, adj, dropout=dropout, normalize=normalize, Euler=Euler) for _ in range(nhl)]
    out_layer = [nn.Linear(hidden_size, num_classes, bias=True)]
    model = nn.Sequential(*in_layer, *feature_layer, *out_layer)

elif args.model == 'odeGCN':
    input_size = features.shape[1]
    hidden_size = args.hidden
    num_classes = labels.max().item() + 1
    dropout = args.dropout
    normalize = args.normalize
    nhl = args.nHiddenLayers
    Euler = args.Euler
    # rownorm. Is there other normalization layer? "RowNorm(),"  nn.BatchNorm1d(hidden_size),
    in_layer = [nn.Linear(input_size, hidden_size, bias=True), RowNorm(), nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size, bias=True)]
    feature_layer = [ODEBlock(ODEFunc(hidden_size, adj, dropout=dropout))]
    out_layer = [nn.Linear(hidden_size, num_classes, bias=True)]

    model = nn.Sequential(*in_layer, *feature_layer, *out_layer)

if args.model == 'differential_gcn':
    print('Choose model:' + args.model)
    input_size = features.shape[1]
    hidden_size = args.hidden
    num_classes = labels.max().item() + 1
    dropout = args.dropout
    T = args.T
    time_tick = args.time_tick
    print('T : {}, time tick: {}'.format(T, time_tick))
    t = torch.linspace(0, T, time_tick).float()
    control = True if args.no_control else False

    embedding_layer = [nn.Linear(input_size, hidden_size, bias=True),  nn.Tanh()]#,
                       # nn.Linear(hidden_size, hidden_size, bias=True)]
        # RowNorm(),,
        #               nn.Linear(hidden_size, hidden_size, bias=True)]
    neural_dynamic_layer = [ODEBlock(
        ODEFunc(hidden_size, adj, dropout=dropout, no_control=control),  # OM
        t,
        rtol=args.rtol, atol=args.atol, method=args.method, terminal=True)]  # t is like  continuous depth
    semantic_layer = [nn.Linear(hidden_size, num_classes, bias=True)]
    model = nn.Sequential(*embedding_layer, *neural_dynamic_layer, *semantic_layer)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


# Send to GPU
if args.cuda:
    model.cuda()


def train(ITER, epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    # output = model(features, adj)
    output = model(features)

    # loss_train = 0.0
    # for x in output:
    #     loss_train += F.cross_entropy(x[idx_train], labels[idx_train])
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        # output = model(features, adj)
        output = model(features)

    loss_val = F.cross_entropy(output [idx_val], labels[idx_val])
    acc_val = accuracy(output [idx_val], labels[idx_val])
    print('ITER: {:04d}'.format(ITER + 1),
          'Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    # output = model(features, adj)
    output = model(features)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test.item(), acc_test.item()


if args.dump:
    fname = "results/results_{}.txt".format(datetime.datetime.now().__str__().replace(':', '-'))
    fout = open(fname, "w")
    fout.write(vars(args).__str__()+"\n")
    fout.write("Time\tLoss\tAccuracy\tStep\n")

for ITER in range(args.iter):
    # Train model
    t_start = time.time()
    for epoch in range(args.epochs):
        train(ITER, epoch)
    print("Optimization Finished!")
    t_total = time.time() - t_start
    print("Total time elapsed: {:.4f}s".format(t_total))

    # Testing
    with torch.no_grad():
        loss_test, acc_test = test()
        time_step = 0  # list(model.parameters())[0].item()
        if args.dump:
            fout.write("{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(t_total, loss_test, acc_test, time_step))
            fout.flush()

T_TOTAL = time.time() - T_VERY_BEGINING
sms_str = "DONE!\nTotal time: {:.4f}s;\n".format(T_TOTAL)
print(sms_str)

if args.dump:
    fout.close()
    r = pd.read_csv(fname, delimiter='\t', skiprows=1)
    rmean = r.loc[:, 'Accuracy'].mean()
    rstd = r.loc[:, 'Accuracy'].std()
    rmedian = r.loc[:, 'Accuracy'].median()
    rmin = r.loc[:, 'Accuracy'].min()
    rmax = r.loc[:, 'Accuracy'].max()
    time_step = r.loc[:, 'Step'].mean()
    print(vars(args).__str__())
    print('results: {:.3f}% +/- {:.3f}%, {:.3f}%;'.format(rmean*100, rstd*100, rmedian*100))
    print('Min_Acc: {:.3f}%, Max_Acc: {:.3f}%'.format(rmin*100, rmax*100))
    print('Time_Step: {:.5f};'.format(time_step))

    sms_str += 'Mean_Acc: {:.3f}% +/- {:.3f}%;\nMedian_acc" {:.3f}%;\n'.format(rmean*100, rstd*100, rmedian*100)
    sms_str += 'Min_Acc: {:.3f}%, Max_Acc: {:.3f}%\n'.format(rmin*100, rmax*100)
    sms_str += 'Time_Step: {:.5f};\n'.format(time_step)
    sms_str += ('Settings: ' + vars(args).__str__())

    if args.sms:
        mysms = SMS()
        mysms.send_sms(sms_str)



