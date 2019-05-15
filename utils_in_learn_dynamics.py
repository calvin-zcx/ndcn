import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import networkx as nx
from networkx.algorithms import community
import matplotlib.cm as cm


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def visualize(N, x0, xt, figname, title ='Dynamics in Complex Network', dir='png_learn_dynamics', zmin=None, zmax=None):
    """
    :param N:   N**2 is the number of nodes, N is the pixel of grid
    :param x0:  initial condition
    :param xt:  states at time t to plot
    :param figname:  figname , numbered
    :param title: title in figure
    :param dir: dir to save
    :param zmin: ax.set_zlim(zmin, zmax)
    :param zmax: ax.set_zlim(zmin, zmax)
    :return:
    """
    if zmin is None:
        zmin = x0.min()
    if zmax is None:
        zmax = x0.max()
    fig = plt.figure()  # figsize=(12, 4), facecolor='white'
    fig.tight_layout()
    x0 = x0.detach()
    xt = xt.detach()
    ax = fig.gca(projection='3d')
    ax.cla()
    # ax.set_title(title)
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
    surf = ax.plot_surface(X, Y, xt.numpy().reshape((N, N)), cmap='rainbow',
                           linewidth=0, antialiased=False, vmin=zmin, vmax=zmax)
    ax.set_zlim(zmin, zmax)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    fig.savefig(dir+'/'+figname+".png", transparent=True)
    fig.savefig(dir+'/'+figname + ".pdf", transparent=True)

    # plt.draw()
    plt.pause(0.001)
    plt.close(fig)


def visualize_graph_matrix(G, title, dir=r'figure/network'):
    A = nx.to_numpy_array(G)
    # plt.pcolormesh(B)
    fig = plt.figure()  # figsize=(12, 4), facecolor='white'
    # plt.title(title)
    fig.tight_layout()
    plt.imshow(A, cmap='Greys')  # ''YlGn')
    # plt.pcolormesh(A)
    plt.show()

    fig.savefig(dir + '/' + title + ".png", transparent=True)
    fig.savefig(dir + '/' + title + ".pdf", transparent=True)


def zipf_smoothing(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    """
    A_prime = A + np.eye(A.shape[0])
    out_degree = np.array(A_prime.sum(1), dtype=np.float32)
    int_degree = np.array(A_prime.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A_prime @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def normalized_plus(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D ^-1/2 * ( A + I ) * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ (A + np.eye(A.shape[0])) @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def normalized_laplacian(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 * ( D - A ) * D^-1/2 = I - D^-1/2 * ( A ) * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.eye(A.shape[0]) - np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def normalized_adj(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 *  A   * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator

def grid_8_neighbor_graph(N):
    """
    Build discrete grid graph, each node has 8 neighbors
    :param n:  sqrt of the number of nodes
    :return:  A, the adjacency matrix
    """
    N = int(N)
    n = int(N ** 2)
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


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()
        self.val = None
        self.avg = 0

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_batch(true_y, t, data_size, batch_time, batch_size, device):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    # s: 20
    batch_y0 = true_y[s]  # (M, D) 500*1*2
    batch_y0 = batch_y0.squeeze() # 500 * 2
    batch_t = t[:batch_time]  # (T) 19
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)
    # (T, M, D) 19*500*1*2   from s and its following batch_time sample
    batch_y = batch_y.squeeze()  # 19 * 500 * 2
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def torch_sensor_to_torch_sparse_tensor(mx):
    """ Convert a torch.tensor to a torch sparse tensor.
    :param torch tensor mx
    :return: torch.sparse
    """
    index = mx.nonzero().t()
    value = mx.masked_select(mx != 0)
    shape = mx.shape
    return torch.sparse.FloatTensor(index, value, shape)


def test():
    a = torch.tensor([[2,0,3], [0,1,-1]]).float()
    print(a)
    b = torch_sensor_to_torch_sparse_tensor(a)
    print(b.to_dense())
    print(b)


def generate_node_mapping(G, type=None):
    """
    :param G:
    :param type:
    :return:
    """
    if type == 'degree':
        s = sorted(G.degree, key=lambda x: x[1], reverse=True)
        new_map = {s[i][0]: i for i in range(len(s))}
    elif type == 'community':
        cs = list(community.greedy_modularity_communities(G))
        l = []
        for c in cs:
            l += list(c)
        new_map = {l[i]:i for i in range(len(l))}
    else:
        new_map = None

    return new_map


def networkx_reorder_nodes(G, type=None):
    """
    :param G:  networkX only adjacency matrix without attrs
    :param nodes_map:  nodes mapping dictionary
    :return:
    """
    nodes_map = generate_node_mapping(G, type)
    if nodes_map is None:
        return G
    C = nx.to_scipy_sparse_matrix(G, format='coo')
    new_row = np.array([nodes_map[x] for x in C.row], dtype=np.int32)
    new_col = np.array([nodes_map[x] for x in C.col], dtype=np.int32)
    new_C = sp.coo_matrix((C.data, (new_row, new_col)), shape=C.shape)
    new_G = nx.from_scipy_sparse_matrix(new_C)
    return new_G





def test_graph_generator():
    n = 400
    m = 5
    seed = 0
    # G = nx.barabasi_albert_graph(n, m, seed)
    G = nx.random_partition_graph([100, 100, 200], .25, .01)
    sizes = [10, 90, 300]
    probs = [[0.25, 0.05, 0.02],
             [0.05, 0.35, 0.07],
             [0.02, 0.07, 0.40]]
    G = nx.stochastic_block_model(sizes, probs, seed=0)

    G = nx.newman_watts_strogatz_graph(400, 5, 0.5)

    A = nx.to_numpy_array(G)
    print(A)
    plt.pcolormesh(A)
    plt.show()

    s = sorted(G.degree, key=lambda x: x[1], reverse=True)
    # newmap = {s[i][0]:i for i in range(len(s))}
    # H= nx.relabel_nodes(G,newmap)
    # newmap = generate_node_mapping(G, type='community')
    # H = networkX_reorder_nodes(G, newmap)
    H = networkx_reorder_nodes(G, 'community')

    # B = nx.to_numpy_array(H)
    # # plt.pcolormesh(B)
    # plt.imshow(B)
    # plt.show()

    visualize_graph_matrix(H)


if __name__ == '__main__':
    test_graph_generator()
