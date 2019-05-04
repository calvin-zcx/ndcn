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


parser = argparse.ArgumentParser('Heat Diffusion Dynamics on Grid demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=100)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')

parser.add_argument('--npix', type=int, default=20)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


N = args.npix
n = N**2

def grid_8_neighboer_graph(N):
    """
    Build discrete grid graph, each node has 8 neighbors
    :param n:  sqrt of the number of nodes
    :return:  A, the adjacency matrix
    """
    n = N ** 2
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    A = torch.zeros(n, n)
    for x in range(N):
        for y in range(N):
            index = x * N + y
            for i in range(len(dx)):
                newx = x + dx[i]
                newy = y + dy[i]
                if N > newx >= 0 and N > newy >= 0:
                    index2 = newx * N + newy
                    A[index, index2] = 1
    return A.float()


# Laplacian Matrix L for heat diffusion
A = grid_8_neighboer_graph(N)
D = torch.diag(A.sum(1))
L = (D - A)

def zipf_smoothing(A):
    """
    Input A: ndarray
    :return:  #  (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    """
    A_prime = A + np.eye(A.shape[0])
    out_degree = np.array(A_prime.sum(1), dtype=np.float32)
    int_degree = np.array(A_prime.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A_prime @ np.diag(int_degree_sqrt_inv)
    return mx_operator

# test = zipf_smoothing(np.array([[1,1], [0, 1]]))


OM = zipf_smoothing(A.numpy())
OM = torch.tensor(OM)

# Initial Value
x0 = torch.zeros(N, N)
x0[1:5, 1:5] = 5
x0[9:15, 9:15] = 10
x0[1:5, 7:13] = 7
x0 = x0.view(-1,1).float()
energy = x0.sum()
# Analysis solution of X' = LX
R,V = torch.eig(-L, eigenvectors=True)
r = R[:,0].view(-1,1)  # torch.mm(L, V) = torch.mm(V, Lambda)

c = torch.mm(V.t(), x0)   # V^-1 = V.t() for real symmetric matrix
solutionList =  [] # [x0.t()]
for t in torch.linspace(0., 5., args.data_size):
    xt = torch.mm(V, torch.exp(t*r)*c)
    energy = xt.sum()
    solutionList.append(xt.t())
    if args.viz:
        plt.imshow(xt.view(N,N))
        # plt.colorbar()
        plt.show()
        # fig.savefig('png/{:03d}'.format(itr))
        # plt.draw()
        plt.pause(0.001)
solution_analysis = torch.stack(solutionList)
print(solution_analysis.shape)


class HeatDiffusion(nn.Module):
    # In this code, row vector: y'^T = y^T A^T      textbook format: column vector y' = A y
    def __init__(self,  L,  k=1):
        super(HeatDiffusion, self).__init__()
        self.L = -L   # Diffusion operator
        self.k = k   # heat capacity

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dX(t)/dt = k * L*X
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        f = torch.mm(self.L, x.t())
        return f.t() * self.k


t = torch.linspace(0., 5., args.data_size) # args.data_size)
with torch.no_grad():
    solution_numerical = odeint(HeatDiffusion(L, 1), x0.t(), t, method='dopri5') # shape: 1000 * 1 * 2
    print(solution_numerical.shape)


print(F.l1_loss(solution_numerical, solution_analysis))

true_y = solution_numerical.to(device)
true_y0 = x0.t().to(device)
L = L.to(device)
OM = OM.to(device)

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    # s: 20
    batch_y0 = true_y[s]  # (M, D) 500*1*2
    batch_y0 = batch_y0.squeeze() # 500 * 2
    batch_t = t[:args.batch_time]  # (T) 19
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)
    # (T, M, D) 19*500*1*2   from s and its following batch_time sample
    batch_y = batch_y.squeeze() # 19 * 500 * 2
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('heat_png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    fig.tight_layout()
    # plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-2, 5)
        # ax_traj.legend()
        # plt.show()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 5)
        ax_phase.set_ylim(-2, 5)
        # plt.show()


        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 5)
        ax_vecfield.set_ylim(-2, 5)
        plt.show()

        fig.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, A, dropout=0.0):
        super(ODEFunc, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A
        # self.nfe = 0
        self.input_layer = nn.Linear (input_size, hidden_size)
        self.outut_layer = nn.Linear(hidden_size, output_size)
        self.wt = nn.Linear(input_size, input_size)
        # if timevarying:
        #     self.linear = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, t, x): # How to use t?
        # self.nfe += 1
        x = self.wt(x)
        x = torch.mm(x, self.A)
        #
        x = torch.tanh(x)
        #x = self.outut_layer(x)
        # x= self.dropout_layer(x)  # drop out for input
        # f = row_normalization(f)
        # f = self.batchnorm(f)
        # f = F.relu(f)  # !!!!! Not use relu seems doesn't  matter!!!!!!
        return x

# in_layer = [nn.Linear(input_size, hidden_size, bias=True), RowNorm(), nn.ReLU(inplace=True),
#                 nn.Linear(hidden_size, hidden_size, bias=True)]
# feature_layer = [ODEFunc(400, 10, 400, L)]
# out_layer = [ ]
#
# model = nn.Sequential(*in_layer, *feature_layer, *out_layer)

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0
    # in_layer = nn.Linear(400, 400).to(device)
    model = ODEFunc(400, 10, 400, OM.float()).to(device)
    # optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    # params = (list(in_layer.parameters()) + list(model.parameters()))
    params = model.parameters()
    optimizer = optim.Adam(params, lr=1e-2, weight_decay=1e-3)

    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch() # batch_y0: 20*1*2  batch_t:10  batch_y: 10*20*1*2
        # batch_y0 = true_y0
        # batch_t = t
        # batch_y = true_y

        pred_y = odeint(model, batch_y0, batch_t, method='dopri5' ) # 'dopri5'
        # loss = torch.mean(torch.abs(pred_y - batch_y))
        # loss = F.mse_loss(pred_y, batch_y)
        loss = F.l1_loss(pred_y, batch_y)

        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(model, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                # visualize(true_y, pred_y, model, ii)
                ii += 1

        end = time.time()
