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

def CLR(affinity, num_category, lambda_c=1, iter_num=40):
    """
    
    Arguments
    ---------
    affinity : adjacency matrix of data
    num_category : number of clusters
    lambda_c : parameter which controls the degree of laplacian rank constrained  
    iter_num : max_iteration
    Returns
    -------
    S : graph constructed by CLR(n*n)
    ans : clustering results (n dim vector)
    """
    
    
    eps = 1e-8
    num_data = affinity.shape[0]
    affinity = (affinity + affinity.T)/2
    affinity_built = np.zeros((num_data,num_data))
    laplacian = get_laplacian(affinity, normalization=0)
    lamb, feature = calc_eigen(laplacian, num_category)
    epoch = 0
    while epoch < iter_num:
        epoch += 1
        v = np.zeros((num_data, num_data))
        for i in range(num_data):
            for j in range(num_data):
                v[i, j] = l2_dist(feature[i], feature[j])
            affinity_built[i] = simplex_opt(affinity[i] - lambda_c / 2 * v[i])
        F_old = feature.copy()
        laplacian = get_laplacian(affinity_built, normalization=0)
        lamb, feature = calc_eigen(laplacian, num_category)
        print(lambda_c, lamb[0:num_category].sum(), lamb[0:num_category + 1].sum())
        if np.sum(lamb[0:num_category]) > eps:
            lambda_c = 2 * lambda_c
        else :
            if np.sum(lamb[0:num_category + 1]) < eps:
                lambda_c = lambda_c / 2
                feature = F_old.copy()
            else :
                break  

    _, ans = connected_components(affinity_built)
    if _ != num_category :
        print("Wrong Clustering", _)
    return affinity_built, ans


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
     