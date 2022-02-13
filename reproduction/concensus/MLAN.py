import os
from sklearn import metrics
import numpy as np
import random
import matplotlib.pyplot as plt
from kmeans import kmeans
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
# from my_attempt import Dataset 
from utils import simplex_opt, get_data, CLR_map, l2_dist, acc, calc_eigen,f_norm, get_laplacian
import evaluation
from CAN import CAN, get_gamma
eps = 1e-8
def dist_map(F):
    """
    Arguments
    ---------
    F : matrix which has the size of (n * fn(feature number)) 
    Returns
    -------
    c : distance calculated pair by pair
    """
    n = F.shape[0]
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c[i][j] = l2_dist(F[i], F[j])
    return c

def initial_graph(e, m = 9):
    """
    
    Arguments
    ---------
    e : initial distance matrix
    m : number of neighbors taken into consider
    Returns
    -------
    W : initial graph for MLAN
    """
    
    
    N = e.shape[0]
    idx = np.zeros((N, m + 1))
    for i in range(N):
        idx[i] = np.argsort(e[i])[:m + 1]
    idx = idx.astype(np.int16)
    # print("idx =", idx)
    W = np.zeros((N, N))
    eps = 1e-8
    for i in range(N):
        id = idx[i, 1:m + 1]
        d = e[i, id]
        W[i, id] = (d[m - 1] - d) / (m * d[m - 1] - np.sum(d) + eps)
    return W

def MLAN(data_v, k, gamma = 100, lambda_1 = 1, epochs = 20):
    """
    Arguments
    ---------
    data_v : list of raw data in V views
    k : the number of clusters
    gamma : a hyperparameter in CAN
    lambda_c : parameter which controls the degree of laplacian rank constrained  
    epochs : max_iteration
    Returns
    -------
    G : clustering results (n dim vector)
    """
    
    V = len(data_v)
    N = data_v[0].shape[0]
    alpha = np.ones(V) / V
    W = np.zeros((N, N))
    W_v = []
    print("Initial Ready")
    for v in range(V):
        curW = dist_map(data_v[v])
        W = W + alpha[v] * curW
        W_v.append(curW)
    W = W / V
    m = 9
    S = initial_graph(W, m)
    L = get_laplacian(S, normalization=0)
    _, F = calc_eigen(L, k)
    print("Prepare to CAN")
    for epoch in range(epochs):
        W = np.zeros((N,N))
        for v in range(V):
            alpha[v] = 0.5 / np.sqrt(l2_dist(alpha[v] * W_v[v], S))
            W += alpha[v] * W_v[v]
        d = dist_map(F) * lambda_1 + W

        for i in range(N):
            S[i] = simplex_opt(-1.0/(2 * gamma) * d[i])
        L = get_laplacian(S, normalization=0)
        F_old = F.copy()
        lamb, F = calc_eigen(L, k)
        print("epoch:", epoch, lambda_1, lamb[0:k].sum(), lamb[0:k + 1].sum())
        print(np.sum(lamb < eps))
        if np.sum(lamb[0:k]) > eps:
            lambda_1 = 2 * lambda_1
        else :
            if np.sum(lamb[0:k + 1]) < eps:
                lambda_1 = lambda_1 / 2
                F = F_old.copy()
            else :
                break
    _, G = connected_components(S)
    if _ != k :
        print("Wrong Clustering", _)
    return G

filename = ""
if __name__ == "__main__":
    Dataset = "Mfeat"
    Dataset = "100leaves"
    data_v, label, k = get_data(dataset = Dataset)
    V = len(data_v)
    lambda_1 = 1
    gamma = 0
    for v in range(V):
        gamma += get_gamma(data_v[v], m = 9)
    gamma /= V
    ans = MLAN(data_v, k, gamma = gamma)
        # f = open("acc1.txt", "a+")
        # print("CLR:", evaluation.clustering(ans, label), file = f)    
    print(evaluation.clustering(ans, label))
    for i in range(k):
        print("Cluster_num" + str(i) + ":", (ans == i).sum())
        # f.close()
        
    # for ww in range(0, 6):
    #     data = data_v[ww].copy()
    #     gamma = get_gamma(data, m = 9)
    #     print(gamma)
    #     S, ans = CAN(data, k, lambda_1=lambda_1)
