import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq as ode


class Propagation:
    def __init__(self, A):
        """
        :param A: supposed to be a  scipy.sparse.csr_matrix()
        """
        self.A = A
        print(str(type(A)))

    def number_of_self_loops(self):
        """
        :return: the number of self-loops in A
        """
        return np.diagonal(self.A.toarray()).sum()

    # def add_self_loop(self):
    #     """
    #     Add self loop to the Adjacency matrix
    #     :return:  A + I
    #     """
    #     return self.A + sp.eye(self.A.shape[0])

    def row_normalization(self):
        """Row-normalize sparse matrix
           :return: D^-1 * A
        """
        out_degree = np.array(self.A.sum(1), dtype=np.float32)
        r_inv = np.power(out_degree, -1, where=(out_degree != 0))
        mx_operator = self.A.multiply(r_inv)
        return mx_operator

    def random_walk(self):
        """See row_normalization, equivalent
           :return:  D^-1 * A
        """
        return self.row_normalization()

    def normalized_laplacian(self):
        """
                :return:  #  I  -  (D )^-1/2 * ( A ) * (D )^-1/2
        """
        out_degree = np.array(self.A.sum(1), dtype=np.float32)
        int_degree = np.array(self.A.sum(0), dtype=np.float32)

        out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
        int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
        mx_operator = sp.eye(self.A.shape[0]) - \
                      sp.csr_matrix(out_degree_sqrt_inv).multiply(self.A).multiply(int_degree_sqrt_inv)
        return mx_operator

    def laplacian(self):
        """
                :return:  #  A - D
        """
        adj = sp.coo_matrix(self.A)
        row_sum = np.array(adj.sum(1)).flatten()
        d_mat = sp.diags(row_sum)
        return (adj - d_mat ).tocoo()

        # out_degree = np.array(self.A.sum(1), dtype=np.float32)
        # int_degree = np.array(self.A.sum(0), dtype=np.float32)
        #
        # out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
        # int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
        # mx_operator = sp.eye(self.A.shape[0]) - \
        #               sp.csr_matrix(out_degree_sqrt_inv).multiply(self.A).multiply(int_degree_sqrt_inv)
        # return mx_operator

    def zipf_smoothing(self):
        """
        :return:  #  (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        """
        assert self.number_of_self_loops() == 0, r"The adjacency matrix has self-loops"

        A_prime = self.A + sp.eye(self.A.shape[0])
        out_degree = np.array(A_prime.sum(1), dtype=np.float32)
        int_degree = np.array(A_prime.sum(0), dtype=np.float32)

        out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
        int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
        mx_operator = sp.csr_matrix(out_degree_sqrt_inv).multiply(A_prime).multiply(int_degree_sqrt_inv)
        return mx_operator  ## - sp.eye(self.A.shape[0])

    def zipf_smoothing_alpha(self, alpha=0.5):
        """
        :return:  #  (aI  + (1-a)D)^-1/2 * ( a * I +  (1-a) * A) * (aI  + (1-a)D)^-1/2
        """
        # assert self.number_of_self_loops() == 0, r"The adjacency matrix has self-loops"
        A_prime = alpha * sp.eye(self.A.shape[0]) + (1 - alpha) * self.A
        out_degree = np.array(A_prime.sum(1), dtype=np.float32)
        int_degree = np.array(A_prime.sum(0), dtype=np.float32)

        out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
        int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
        mx_operator = sp.csr_matrix(out_degree_sqrt_inv).multiply(A_prime).multiply(int_degree_sqrt_inv)
        return mx_operator  ## - sp.eye(self.A.shape[0])

    def zipf_smoothing_prime(self):
        """
        :return:  #  (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2  - I
        """
        # assert self.number_of_self_loops() == 0, r"The adjacency matrix has self-loops"

        A_prime = self.A + sp.eye(self.A.shape[0])
        out_degree = np.array(A_prime.sum(1), dtype=np.float32)
        int_degree = np.array(A_prime.sum(0), dtype=np.float32)

        out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
        int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
        mx_operator = sp.csr_matrix(out_degree_sqrt_inv).multiply(A_prime).multiply(int_degree_sqrt_inv) - sp.eye(self.A.shape[0])
        return mx_operator

    def first_order_gcn(self):
        """
            :return:  #  I + (D )^-1/2 * ( A ) * (D )^-1/2
        """
        out_degree = np.array(self.A.sum(1), dtype=np.float32)
        int_degree = np.array(self.A.sum(0), dtype=np.float32)

        out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
        int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
        mx_operator = sp.eye(self.A.shape[0]) + sp.csr_matrix(out_degree_sqrt_inv).multiply(self.A).multiply(int_degree_sqrt_inv)
        return mx_operator

    def residual_smoothing(self, delta):
        """
        :return:  #  (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        """
        assert self.number_of_self_loops() == 0, r"The adjacency matrix has self-loops"

        A_prime = delta * self.A + sp.eye(self.A.shape[0])
        out_degree = np.array(A_prime.sum(1), dtype=np.float32)
        int_degree = np.array(A_prime.sum(0), dtype=np.float32)

        out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
        int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
        mx_operator = sp.csr_matrix(out_degree_sqrt_inv).multiply(A_prime).multiply(int_degree_sqrt_inv)
        return mx_operator

    def __aug_normalized_adjacency__(self):
        """
        Codes from SGC, Supposed to be == zipf_smoothing()
        For test
        :return: #  (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        """
        adj = self.A
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()








