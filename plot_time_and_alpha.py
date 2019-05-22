import os
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import networkx as nx
from networkx.algorithms import community
import matplotlib.cm as cm
import seaborn as sns
from utils_in_learn_dynamics import *
from utils import  *


def extract_results(data='citeseer'):
    filepath = 'results/{}/output_{}_time_and_alpha.txt'.format(data, data)
    dumppath = 'results/{}/output_{}_time_and_alpha.npy'.format(data, data)
    results = np.empty([0, 5])
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            # print("Line {}: {}".format(cnt, line))
            if line[0:7] == r'results':
                v = line.split()
                vmean = float(line.split()[1].strip('%;,'))
                vstd = float(line.split()[3].strip('%;,'))
                vmed = float(line.split()[4].strip('%;,'))
            if line[0:7] == r'Min_Acc':
                v = line.split()
                vmin= float(line.split()[1].strip('%;,'))
                vmax = float(line.split()[3].strip('%;,'))
                results = np.vstack((results, [vmean, vstd, vmed, vmin, vmax]))

    if data == "cora" or data == "pubmed":
        r1_1_5 = results[:66,:]
        r_05_09 = results[66:, :]
        results = np.vstack((r_05_09, r1_1_5))
    print('size of results: {}'.format(results.shape))
    np.save(dumppath, results)
    test_r = np.load(dumppath)
    print('size of results: {}'.format(test_r.shape))


def extract_results_for_pubmed():
    # for part results. DELETED
    filepath = 'results/pubmed/output_pubmed_time_and_alpha.txt'
    filepath2 = 'results/pubmed/output_pubmed_time_and_alpha_part.txt'
    dumppath = 'results/pubmed/output_pubmed_time_and_alpha.npy'
    results = np.empty([0, 5])
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            # print("Line {}: {}".format(cnt, line))
            if line[0:7] == r'results':
                v = line.split()
                vmean = float(line.split()[1].strip('%;,'))
                vstd = float(line.split()[3].strip('%;,'))
                vmed = float(line.split()[4].strip('%;,'))
            if line[0:7] == r'Min_Acc':
                v = line.split()
                vmin= float(line.split()[1].strip('%;,'))
                vmax = float(line.split()[3].strip('%;,'))
                results = np.vstack((results, [vmean, vstd, vmed, vmin, vmax]))
    results2 = np.empty([0, 5])
    with open(filepath2) as fp:
        for cnt, line in enumerate(fp):
            # print("Line {}: {}".format(cnt, line))
            if line[0:7] == r'results':
                v = line.split()
                vmean = float(line.split()[1].strip('%;,'))
                vstd = float(line.split()[3].strip('%;,'))
                vmed = float(line.split()[4].strip('%;,'))
            if line[0:7] == r'Min_Acc':
                v = line.split()
                vmin = float(line.split()[1].strip('%;,'))
                vmax = float(line.split()[3].strip('%;,'))
                results2 = np.vstack((results2, [vmean, vstd, vmed, vmin, vmax]))

    un = results2.shape[0]
    # r1_1_5 = results[:66,:]
    # r_05_09 = results[66:, :]
    assert un <= 55
    results[66:66+un] = results2
    print('size of results: {}'.format(results.shape))
    np.save(dumppath, results)
    test_r = np.load(dumppath)
    print('size of results: {}'.format(test_r.shape))


def plot_acc_time_alpha_3d(data='citeseer', dump=False):
    dumppath = 'results/{}/output_{}_time_and_alpha.npy'.format(data, data)
    r = np.load(dumppath)
    print('size of results: {}'.format(r.shape))
    # [vmean, vstd, vmed, vmin, vmax]
    vmean = r[:, 0].reshape((-1, 11)) # T * alpha
    vstd = r[:, 1].reshape((-1, 11))

    fig = plt.figure()  # figsize=(12, 4), facecolor='white'
    fig.tight_layout()

    ax = fig.gca(projection='3d')
    ax.cla()
    # ax.set_title(title)
    X = np.arange(0, 11)
    Y = np.arange(0, 11)
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
    surf = ax.plot_surface(X, Y, vmean / 100., cmap='rainbow', #cm.coolwarm, #'rainbow',
                           linewidth=0, antialiased=False) #, vmin=zmin, vmax=zmax)
    #ax.set_zlim(zmin, zmax)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('Alpha')
    ax.set_xlim(0, 10)
    ax.set_ylabel('Terminal Time')
    ax.set_ylim(0, 10)
    ax.set_zlabel('Accuracy')
    #ax.set_zlim(-100, 100)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ['{:.1f}'.format(x/10.0) for x in ax.get_xticks()]
    ax.set_xticklabels(labels)

    labels = ['{:.1f}'.format(x / 10.0 + 0.5) for x in ax.get_yticks()]
    ax.set_yticklabels(labels)

    plt.show()
    print('')
    if dump:
        fig.savefig('results/{}/output_{}_time_and_alpha_3d.png'.format(data, data), transparent=True)
        fig.savefig('results/{}/output_{}_time_and_alpha_3d.pdf'.format(data, data), transparent=True)


def plot_acc_time_alpha_2d(data='citeseer', dump=False):
    dumppath = 'results/{}/output_{}_time_and_alpha.npy'.format(data.lower(), data.lower())
    r = np.load(dumppath)
    print('size of results: {}'.format(r.shape))
    # [vmean, vstd, vmed, vmin, vmax]
    vmean = r[:, 0].reshape((-1, 11)) # T * alpha
    vstd = r[:, 1].reshape((-1, 11))

    fig, ax = plt.subplots()
    fig.tight_layout()
    plt.title(data, fontsize=18)
    line_mark = {'cora':'-sk', 'citeseer':'-ok', 'pubmed':'-^k'}
    max_alpha = {'cora': 0, 'citeseer': 8, 'pubmed': 4}
    alpha = max_alpha[data.lower()]
    mark = line_mark[data.lower()]
    ax.set_xlim(0.45, 1.55)
    plt.errorbar(np.arange(0.5, 1.6, 0.1), vmean[:, alpha]/100., yerr=vstd[:, alpha]/100., fmt=mark,
                 linewidth=2, markersize=12) #, color='deepskyblue')
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.xlabel('Terminal Time', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.show()
    if dump:
        fig.savefig('results/{}/output_{}_time_and_alpha_errorbar.png'.format(data.lower(), data.lower()), transparent=True)
        fig.savefig('results/{}/output_{}_time_and_alpha_errorbar.pdf'.format(data.lower(), data.lower()), transparent=True)


def plot_acc_time_alpha_heatmap(data='citeseer', dump=False):
    dumppath = 'results/{}/output_{}_time_and_alpha.npy'.format(data, data)
    r = np.load(dumppath)
    print('size of results: {}'.format(r.shape))
    # [vmean, vstd, vmed, vmin, vmax]
    vmean = r[:, 0].reshape((-1, 11)) # T * alpha
    vstd = r[:, 1].reshape((-1, 11))

    fig, ax = plt.subplots()
    fig.tight_layout()

    sns.axes_style("white")
    sns.heatmap(vmean.T, annot=True, fmt='.1f', cmap='rainbow') #, cmap='rainbow') #, cmap="YlGnBu") # center=.2)
    # plt.errorbar(np.arange(0.5, 1.6, 0.1), vmean[:, 6]/100., yerr=vstd[:, 6]/100., fmt='o', color='black',
    #              ecolor='lightgray', elinewidth=3, capsize=0)
    # plt.errorbar(np.arange(0.5, 1.6, 0.1), vmean[:, 6]/100., yerr=vstd[:, 6]/100., fmt='-ok')
    # plt.xlabel('Terminal Time', fontsize=18)
    # plt.ylabel('Accuracy', fontsize=18)
    # labels = [item.get_text() for item in ax.get_xticklabels()]

    labels = ['{:.1f}'.format(float(x.get_text()) / 10.0 + 0.5) for x in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    labels = ['{:.1f}'.format(float(y.get_text()) / 10.0) for y in ax.get_yticklabels()]
    ax.set_yticklabels(labels)

    plt.xlabel('Terminal Time', fontsize=18)
    plt.ylabel('Alpha', fontsize=18)
    plt.show()

    if dump:
        fig.savefig('results/{}/output_{}_time_and_alpha.png'.format(data, data), transparent=True)
        fig.savefig('results/{}/output_{}_time_and_alpha.pdf'.format(data, data), transparent=True)


def plot_acc_time_alpha( ):

    dumppath = 'results/citeseer/output_citeseer_time_and_alpha.npy'
    r_citeseer = np.load(dumppath)
    print('size of results: {}'.format(r_citeseer.shape))
    # [vmean, vstd, vmed, vmin, vmax]
    vmean_citeseer = r_citeseer[:, 0].reshape((-1, 11)) # T * alpha
    vstd_citeseer = r_citeseer[:, 1].reshape((-1, 11))

    dumppath = '../results/pubmed/output_pubmed_time_and_alpha.npy'
    r_pubmed = np.load(dumppath)
    print('size of results: {}'.format(r_pubmed.shape))
    vmean_pubmed = r_pubmed[:, 0].reshape((-1, 11))  # T * alpha
    vstd_pubmed = r_pubmed[:, 1].reshape((-1, 11))

    # fig, ax = plt.figure()  # figsize=(12, 4), facecolor='white'
    fig, ax = plt.subplots()
    fig.tight_layout()

    # sns.heatmap(vmean/100.)
    # plt.errorbar(np.arange(0.5, 1.6, 0.1), vmean[:, 6]/100., yerr=vstd[:, 6]/100., fmt='o', color='black',
    #              ecolor='lightgray', elinewidth=3, capsize=0)
    plt.errorbar(np.arange(0.5, 1.6, 0.1), vmean_citeseer[:, 6]/100., yerr=vstd_citeseer[:, 6]/100., fmt='-ok', label='Citeseer')
    plt.errorbar(np.arange(0.5, 1.6, 0.1), vmean_pubmed[:, 6]/100., yerr=vstd_pubmed[:, 6]/100., fmt='-^g', label='Pubmed')
    plt.xlabel('Terminal Time', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    legend = ax.legend( shadow=True, fontsize='x-large')  # loc='upper center',

    plt.show()


def plot_matrix(A, dataset):

    G = nx.from_numpy_array(A)
    G = networkx_reorder_nodes(G, 'community')
    Ao = nx.to_numpy_array(G)
    fig = plt.figure()  # figsize=(12, 4), facecolor='white'
    plt.title(dataset)
    fig.tight_layout()
    plt.imshow(Ao, cmap='binary')  # ''YlGn') Greys YlGn
    # plt.pcolormesh(A)
    plt.show()

    # fig.savefig(dir + '/' + title + ".png", transparent=True)
    # fig.savefig(dir + '/' + title + ".pdf", transparent=True)


if __name__ == '__main__':

    # extract_results_for_pubmed()
    extract_results('pubmed')
    # extract_results('cora')
    #plot_acc_time_alpha_2d('pubmed')
    # plot_acc_time_alpha_2d('citeseer')

    # plot_acc_time_alpha( )
    # plot_acc_time_alpha_3d('citeseer')

    dataset = 'Pubmed'  # 'Pubmed' #   #'Cora' #'Citeseer'
    plot_acc_time_alpha_heatmap(dataset, True)
    plot_acc_time_alpha_2d(dataset, True)
    plot_acc_time_alpha_3d(dataset, True)


    # adj, features, labels, idx_train, idx_val, idx_test = load_data_np(dataset)
    # plot_matrix(adj.todense(), dataset)
