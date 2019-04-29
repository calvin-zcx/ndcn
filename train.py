import argparse
import time
import datetime
# import torch
import torch.nn.functional as F
import torch.optim as optim

from models import *

import pandas as pd

# import propagation as prp
# import scipy.sparse as sp
# import numpy as np

from utils import *
from sms import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=-1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
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
                    choices=["DeepGCN", "GCN", "DeepGCN2", "DeepGCN3", "DeepGCN4"],
                    help='model to use.')
parser.add_argument('--iter', type=int, default=1, help='Number of experiments to conduct')
parser.add_argument('--dump', action='store_true', default=False,
                    help='Dump results to time appendix file.')
parser.add_argument('--delta', type=float, default=1.0, help='Scale of signals from neighborhoods')

parser.add_argument('--sms', action='store_true', default=False,
                    help='Send results short message to my Phone.')

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


optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


# Send to GPU
if args.cuda:
    model.cuda()



def train(ITER, epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('ITER: {:04d}'.format(ITER + 1),
          'Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
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
        time_step = list(model.parameters())[0].item()
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

    # np.random.seed(42)
    # a = np.random.randint(2, size=(10, 2))
    # a = np.array([[0,1,1], [1,0,0], [1,0,0]])
    # print(a)
    # A = sp.csr_matrix(a)
    # print(A)
    # P = prp.Propagation(A)
    # Ap = P.row_normalization()
    # print(Ap.toarray())
    # print(Ap.sum(1))
    #
    # Ap1 = P.zipf_smoothing()
    # print(Ap1.toarray())
    # print(Ap1.sum(1))
    #
    # Ap2 = P.__aug_normalized_adjacency__()
    # print(Ap2.toarray())
    # print(Ap2.sum(1))

