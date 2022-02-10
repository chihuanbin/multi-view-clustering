# -*- encoding: utf-8 -*-
'''
Filename         :AMGL.py
Description      :
Time             :2022/02/10 23:01:12
Author           :zqyang
Version          :1.0
'''

import os
from sklearn import metrics
import numpy as np
import random
import matplotlib.pyplot as plt
from CLR import calc_dist
from kmeans import kmeans
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components 
from utils import get_laplacian, simplex_opt, get_data, CLR_map, l2_dist, acc, calc_eigen,f_norm 
from CLR import CLR
from sklearn.metrics.cluster import normalized_mutual_info_score
import evaluation

def dist_map(F):
    
    n = F.shape[0]
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c[i][j] = l2_dist(F[i], F[j])
    return c
eps = 1e-8
Dataset = "100leaves"
Dataset = "Caltech101-7"
Dataset = "Mfeat"
data_v, label, k = get_data(dataset=Dataset)

def AMGL(W, k, lambda_1 = 1):
    """

    Arguments
    ---------
    W : list of NxN graphs in V views
    k : the number of categories.
    Returns
    -------
    ans : prediction results.
    """

    epoch = 0
    V = len(W)
    alpha = np.ones(V) / V
    W_c = np.zeros((n, n))
    L = []
    for v in range(V):
        W_c += alpha[v] * W[v]   
        L.append(get_laplacian(W[v], normalization=0))   
    W_c /= np.sum(alpha) 
    L_c = get_laplacian(W_c, normalization=0)
    _, F = calc_eigen(L_c, k)
    for epoch in range(20):
        for v in range(V):
            alpha[v] = 1/(2 * np.sqrt(np.trace(F.T @ L[v] @ F)))
        W_c = np.zeros((n, n))
        for v in range(V):
            W_c += alpha[v] * W[v]  
        L_c = get_laplacian(W_c, normalization=0)
        _, F = calc_eigen(L_c, k)
        _, ans = kmeans(F, k)
        print(evaluation.clustering(ans, label))
        for i in range(k):
            print("Cluster_num" + str(i) + ":", (ans == i).sum())
    return ans

n = label.shape[0]
W_v = []
V = len(data_v)
for i in range(V):
    W = CLR_map(data_v[i])
    W_v.append(W)
lambda_c = 1
_ = AMGL(W_v, k)
