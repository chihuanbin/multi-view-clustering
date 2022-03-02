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
    num_data = F.shape[0]
    map = np.zeros((num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            map[i][j] = l2_dist(F[i], F[j])
    return map
eps = 1e-8


def AMGL(W, label):
    """

    Arguments
    ---------
    W : list of NxN graphs in num_view views
    label : the label of each data.
    Returns
    -------
    ans : prediction results.
    """

    epoch = 0
    num_view = len(W)
    num_data = label.shape[0]
    alpha = np.ones(num_view) / num_view
    W_consensus = np.zeros((num_data, num_data))
    num_category = label.max() - label.min()
    Laplacian = []
    for v in range(num_view):
        W_consensus += alpha[v] * W[v]   
        Laplacian.append(get_laplacian(W[v], normalization=0))   
    W_consensus /= np.sum(alpha) 
    L_consensus = get_laplacian(W_consensus, normalization=0)
    _, Feature = calc_eigen(L_consensus, num_category)
    for epoch in range(20):
        for v in range(num_view):
            alpha[v] = 1/(2 * np.sqrt(np.trace(Feature.T @ Laplacian[v] @ Feature)))
        W_consensus = np.zeros((num_data, num_data))
        for v in range(num_view):
            W_consensus += alpha[v] * W[v]  
        L_consensus = get_laplacian(W_consensus, normalization=0)
        _, Feature = calc_eigen(L_consensus, num_category)
        _, ans = kmeans(Feature, num_category)
        print(evaluation.clustering(ans, label))
        for i in range(num_category):
            print("Cluster_num" + str(i) + ":", (ans == i).sum())
    return ans
if __name__ == "__main__":
    Dataset = "100leaves"
    Dataset = "Caltech101-7"
    Dataset = "Mfeat"
    data_v, label, k = get_data(dataset=Dataset)
    n = label.shape[0]
    W_v = []
    V = len(data_v)
    for i in range(V):
        W = CLR_map(data_v[i])
        W_v.append(W)
    _ = AMGL(W_v, k)
