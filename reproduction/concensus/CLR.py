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

def eu_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def l2_dist(x, y):
    return np.sum((x - y) ** 2)


def calc_dist(x, y, method="eu"):
    if method == "eu":
        return eu_dist(x, y)

def CLR(W, k, lambda_c=1, iter_num=40):
    """
    
    Arguments
    ---------
    W : adjacency matrix of data
    k : number of clusters
    lambda_c : parameter which controls the degree of laplacian rank constrained  
    
    Returns
    -------
    S : graph constructed by CLR(n*n)
    G : clustering results (n dim vector)
    """
    
    
    eps = 1e-8
    n = W.shape[0]
    W = (W + W.T)/2
    S = np.zeros((n,n))
    L = get_laplacian(W, normalization=0)
    lamb, F = calc_eigen(L, k)
    epoch = 0
    while epoch < iter_num:
        epoch += 1
        v = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                v[i, j] = l2_dist(F[i], F[j])
            S[i] = simplex_opt(W[i] - lambda_c / 2 * v[i])
        F_old = F.copy()
        L = get_laplacian(S, normalization=0)
        lamb, F = calc_eigen(L, k)
        print(lambda_c, lamb[0:k].sum(), lamb[0:k + 1].sum())
        if np.sum(lamb[0:k]) > eps:
            lambda_c = 2 * lambda_c
        else :
            if np.sum(lamb[0:k + 1]) < eps:
                lambda_c = lambda_c / 2
                F = F_old.copy()
            else :
                break  

    _, G = connected_components(S)
    if _ != k :
        print("Wrong Clustering", _)
    return S, G


filename = ""
if __name__ == "__main__":
    data_v, label, k = get_data(dataset = "Caltech101-7")
    m = 4
    for i in range(5,6):
        data = data_v[i].copy()
        print(data.shape)  
        n = data.shape[0]
        e = np.zeros((n, n))    
        for i in range(n):
            for j in range(n):
                e[i, j] = l2_dist(data[i], data[j])
        print("calc_dis finished")
        idx = np.zeros((n, m + 1))

        for i in range(n):
            idx[i] = np.argsort(e[i])[:m + 1]

        idx = idx.astype(np.int16)
        W = np.zeros((n, n))
        eps = 1e-8
        for i in range(n):
            id = idx[i, 1:m + 1]
            d = e[i, id]
            W[i, id] = (d[m - 1] - d + eps / m) / (m * d[m - 1] - np.sum(d) + eps)

        S, ans = CLR(W, k)
        for i in range(k):
            print("Cluster_num" + str(i) + ":", (ans == i).sum())
        print(evaluation.clustering(ans, label))
     