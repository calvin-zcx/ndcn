import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[0.9, 1.8]])  # shape: 1 * 2
# true_y0 = torch.tensor([[0.1, 0.2]])

t = torch.linspace(0., 25., args.data_size) # shape: 1000

t = torch.linspace(-5., 5., args.data_size) # shape: 1000

# true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]) # shape: 2 * 2

true_A = torch.tensor([[2.0/3, -4.0/3], [1, -1]])

# true_A = torch.tensor([[0, 1], [-1, 0.85]])
#
# true_A = torch.tensor([[0, 1], [-1, -0.45]])


class Lambda(nn.Module):
    # In this code, row vector: y'^T = y^T A^T      textbook format: column vector y' = A y
    def forward(self, t, y):
        # return F.leaky_relu(torch.mm(y**3, true_A))
        # return F.tanh(torch.mm(y ** 3, true_A))
        # return torch.mm(y ** 3, true_A)
        # return torch.mm(y ** 3, true_A)

        x = torch.tensor([[y[0, 0], y[0, 0] * y[0, 1]], [y[0, 0] * y[0, 1], y[0, 1]]])
        y1 = torch.diag(torch.mm(x, true_A.t()))
        # y1 = F.tanh(y1)
        # result = y1/torch.sum(y1)
        # result = F.normalize(y1)

        # return torch.mm(y, true_A.t())
        return  y1


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5') # shape: 1000 * 1 * 2
    print(true_y.shape)    # batchsize*1*2
    # my = torch.mean(true_y[:,0,1])
    # vy = torch.std(true_y[:,0,1])
    # true_y[:,0,1] = (true_y[:,0,1] - my) / vy
    #
    # mx = torch.mean(true_y[:, 0, 0])
    # vx = torch.std(true_y[:, 0, 0])
    # true_y[:, 0, 1] = (true_y[:, 0, 1] - mx) / vx

# test ground truth
plt.plot(true_y[:,0,0].numpy(), true_y[:,0,1].numpy(), '-o')
plt.show()

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    # s: 20
    batch_y0 = true_y[s]  # (M, D) 500*1*2
    batch_y0 = batch_y0.squeeze() # 500 * 2
    batch_t = t[:args.batch_time]  # (T) 19
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)
    # (T, M, D) 19*500*1*2   from s and its following batch_time sample
    batch_y = batch_y.squeeze() # 19 * 500 * 2
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
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

    def __init__(self):
        super(ODEFunc, self).__init__()

        # self.net = nn.Sequential(
        #     nn.Linear(2, 10),
        #     #nn.BatchNorm1d(10),
        #     nn.Tanh(),   # nn.ReLU(), #
        #     nn.Linear(10, 2) #,
        #     # nn.BatchNorm1d(2),
        # )
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )
        # self.scale = nn.Parameter(torch.FloatTensor([1]))  # [0.01]))  # np.random.rand(1) *
        # self.bias = nn.Parameter(torch.FloatTensor([0]))

        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         # nn.init.normal_(m.weight, mean=0, std=0.1)
        #         # nn.init.constant_(m.bias, val=0)
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.normal_(m.bias, mean=1, std=1)
        #         # nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # z = y*1.0
        # z[:,0,1] =  y[:,0,0] * y[:,0,1]
        # A = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        result = self.net(y) #* self.scale + self.bias
        return result #self.net(y) * self.scale + self.bias  #*self.scale    y = batchsize * 1 * 2


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

    func = ODEFunc()
    # optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    optimizer = optim.Adam(func.parameters(), lr=1e-2, weight_decay=1e-3)

    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch() # batch_y0: 20*1*2  batch_t:10  batch_y: 10*20*1*2
        # batch_y0 = true_y0
        # batch_t = t
        # batch_y = true_y
        pred_y = odeint(func, batch_y0, batch_t, method='rk4' ) # 'dopri5'
        # loss = torch.mean(torch.abs(pred_y - batch_y))
        # loss = F.mse_loss(pred_y, batch_y)
        loss = F.l1_loss(pred_y, batch_y)

        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()
