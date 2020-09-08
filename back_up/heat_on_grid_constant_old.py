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
parser.add_argument('--data_size', type=int, default=10)
parser.add_argument('--batch_time', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=10,
                    help='Number of hidden units.')
parser.add_argument('--npix', type=int, default=20)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    makedirs('png_heat')


def visualize(N, Z, itr, dir='png_heat'):
    if args.viz:
        fig = plt.figure()  # figsize=(12, 4), facecolor='white'
        # ax_traj = fig.add_subplot(131, frameon=False)
        # ax_phase = fig.add_subplot(132, frameon=False)
        # ax_vecfield = fig.add_subplot(133, frameon=False)
        ax = fig.gca(projection='3d')
        fig.tight_layout()
        ax.cla()
        ax.set_title('Heat Diffusion')
        X = np.arange(0, N)
        Y = np.arange(0, N)
        X, Y = np.meshgrid(X, Y)  # X, Y, Z : 20 * 20
        # R = np.sqrt(X ** 2 + Y ** 2)
        # Z = np.sin(R)
        # fig.set_xlabel('t')
        # ax_traj.set_ylabel('x,y')
        # ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        # ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        # ax_traj.set_xlim(t.min(), t.max())
        # ax_traj.set_ylim(-2, 5)
        # ax.pcolormesh(xt.view(N,N), cmap=plt.get_cmap('hot'))
        surf = ax.plot_surface(X, Y, xt.detach().numpy().reshape((N,N)), cmap='rainbow',
                               linewidth=0, antialiased=False, vmin=0, vmax=25)
        ax.set_zlim(-1, 25)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()
        fig.savefig(dir+'/{:06d}'.format(itr))
        # plt.draw()
        plt.pause(0.001)
        plt.close(fig)

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
# test = zipf_smoothing(np.array([[1,1], [0, 1]]))


OM = zipf_smoothing(A.numpy())
OM = torch.tensor(OM).float()

# Initial Value
x0 = torch.zeros(N, N)
x0[1:5, 1:5] = 25
x0[9:15, 9:15] = 20
x0[1:5, 7:13] = 17
x0 = x0.view(-1,1).float()
energy = x0.sum()
# Analysis solution of X' = LX
R,V = torch.eig(-L, eigenvectors=True)
r = R[:,0].view(-1,1)  # torch.mm(L, V) = torch.mm(V, Lambda)

c = torch.mm(V.t(), x0)   # V^-1 = V.t() for real symmetric matrix
solutionList = [] # [x0.t()]

ii = 0
for t in torch.linspace(0., 5., args.data_size):
    ii += 1
    xt = torch.mm(V, torch.exp(t*r)*c)
    # visualize(N, xt, ii)
    energy = xt.sum()
    solutionList.append(xt.t())

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


t = torch.linspace(0., 5., args.data_size) # args.data_size) # 100 vector
with torch.no_grad():
    solution_numerical = odeint(HeatDiffusion(L, 1), x0.t(), t, method='dopri5') # shape: 1000 * 1 * 2
    print(solution_numerical.shape)

ii = 0
for xt in solution_numerical:
    ii += 1
    print(xt.shape)
    # visualize(N, xt, ii)


print(F.l1_loss(solution_numerical, solution_analysis))

true_y = solution_numerical.squeeze().t().to(device) # 100 * 1 * 400  --squeeze--> 100 * 400 -t-> 400 * 100
true_y0 = x0.to(device) # 400 * 1
L = L.to(device) # 400 * 400
OM = OM.to(device) # 400 * 400


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


class ODEFunc(nn.Module):
    def __init__(self, hidden_size, A, dropout=0.0):
        super(ODEFunc, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A  # N_node * N_node
        # self.nfe = 0
        self.wt = nn.Linear(hidden_size, hidden_size)

    def forward(self, t, x): # How to use t?
        """
        :param t:  end time tick
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        # self.nfe += 1
        x = torch.mm(self.A, x)
        x = self.dropout_layer(x)
        x = self.wt(x)
        # x = torch.tanh(x)
        x = F.relu(x)  # !!!!! Not use relu seems doesn't  matter!!!!!! in theory. Converge faster
        # x = torch.sigmoid(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc, vt):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time_vector = vt # time vector

    def forward(self, x):
        self.integration_time_vector = self.integration_time_vector.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time_vector, rtol=.01, atol=.01)
        # return out[-1]
        return out # 100 * 400 * 10


input_size = true_y0.shape[1]   # y0: 400*1 ,  input_size:1
hidden_size = 20 # args.hidden  # 10 default
dropout = args.dropout # 0 default
num_classes = 1  # 1 for regression
vt = torch.linspace(0., 5., args.data_size)  # args.data_size 100 (100 depth)

# Build model
embedding_layer = [nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(), # nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size, bias=True)]
ODE_layer = [ODEBlock(ODEFunc(hidden_size, OM, dropout=dropout), vt)]
semantic_layer = [nn.Linear(hidden_size, num_classes, bias=True)]
model = nn.Sequential(*embedding_layer, *ODE_layer, *semantic_layer).to(device)


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

    params = model.parameters()
    optimizer = optim.Adam(params, lr=0.01, weight_decay=1e-3)

    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()

        pred_y = model(true_y0)  # 100 * 400 * 1 should be 400 * 100
        pred_y = pred_y.squeeze().t()

        # pred_y = odeint(model, in_layer(batch_y0), in_layer(batch_t), method='dopri5' ) # 'dopri5'
        # loss = torch.mean(torch.abs(pred_y - batch_y))
        # loss = F.mse_loss(pred_y, true_y)
        loss = F.l1_loss(pred_y, true_y)

        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                ii += 1
                pred_y = model(true_y0).squeeze().t() # odeint(model, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                # visualize(true_y, pred_y, model, ii)
        end = time.time()

    with torch.no_grad():
        for ii in range(pred_y.shape[1]):
            xt = pred_y[:,ii].cpu()
            print(xt.shape)
            visualize(N, xt, ii+100)
