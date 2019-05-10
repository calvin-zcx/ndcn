# import numpy as np
# import scipy.sparse as sp
import pickle as pkl
import torch
import sys
from propagation import *
from sklearn.metrics import f1_score


def sparse_csr_matrix_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy.sparse.csr_matrix to a torch sparse tensor.
    :param sparse_mx:  scipy.sparse.csr_matrix
    :return: torch.sparse.
    """
    # Leverage scipy.sparse.coo_matrix for the row, col, and data
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(dataset_name=r"cora", delta=1):
    r"""
     Load citation network datasets from ./data/NAME/ directory, and return
     the train, validation, test features, labels, and underlying networks (or its propagation operator)
    :return:

    Datasets' format:
        ind.dataset_name.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_name.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_name.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_name.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_name.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_name.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_name.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_name.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_name.test.index => the indices of test instances in graph, for the inductive setting as list object.
        All objects above must be saved using python pickle module.

    Args:

    Examples:

    """
    # x: scipy.sparse.csr.csr_matrix
    # y: numpy.ndarray
    # index: list
    # graph: collections.defaultdict
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open("data/{}/ind.{}.{}".format(dataset_name.lower(), dataset_name.lower(), name), "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = np.loadtxt("data/{}/ind.{}.test.index".format(dataset_name.lower(), dataset_name.lower()),
                                  dtype=np.int64)
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_name == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        # Added later
        pass

    # Combine the train and test feature matrices and labels lists into a united one
    features = sp.vstack((allx, tx)).tolil()
# #########################################
#     features = features.toarray()
#     feature_sort_index = np.argsort(np.array(features.sum(0)))
#     """np.sort(b)[-600:].sum() / b.sum()
#         0.84053963
# """
#     features[:, feature_sort_index[-500:]] = 0
#     features = sp.lil_matrix(features)
########################################################################

    features[test_idx_reorder,:] = tx  # features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = ty  # labels[test_idx_range, :]

    # Build graph from a dictionary of list to a sparse matrix for computing
    row_col = [(row, col) for row in graph for col in graph.get(row)]
    adj = sp.csr_matrix((np.ones(len(row_col)), (zip(*row_col))))
    # from directed citation graph to undirected symmetric graph
    adj = adj + adj.T
    adj[adj > 1] = 1

    # Divide data into train, validation, and test datasets
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()

    # Normalize feature matrix
    prp_features = Propagation(features)
    features_normalized = prp_features.row_normalization()

    # Graph convolution operator
    prp_gcn = Propagation(adj)
    adj_gcn = prp_gcn.zipf_smoothing()
    # print('Delta: {}'.format(delta))
    # adj_gcn = prp_gcn.residual_smoothing(delta)
    # adj_gcn = prp_gcn.first_order_gcn()
    # adj_gcn = prp_gcn.normalized_laplacian()  # * (-1.0)
    # adj_gcn = prp_gcn.laplacian()
    # adj_gcn = adj
    # adj_gcn = prp_gcn.zipf_smoothing_prime()

    # From scipy.sparse.csr_matrix to torch.sparse.FloatTensor
    # Warning: feature matrix must be tensor dense one, and the adjacency matrix can be torch.sparse one
    torch_adj_gcn = sparse_csr_matrix_to_torch_sparse_tensor(adj_gcn)
    # torch_features_normalized = sparse_csr_matrix_to_torch_sparse_tensor(features_normalized)
    torch_features_normalized = torch.FloatTensor(features_normalized.todense())
    torch_labels = torch.LongTensor(labels.tolist()).max(1)[1]
    torch_idx_train = torch.LongTensor(idx_train)
    torch_idx_val = torch.LongTensor(idx_val)
    torch_idx_test = torch.LongTensor(idx_test)

    return torch_adj_gcn, torch_features_normalized, torch_labels, torch_idx_train, torch_idx_val, torch_idx_test


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro

