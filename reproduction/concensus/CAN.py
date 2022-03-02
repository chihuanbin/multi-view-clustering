import os
from sklearn import metrics
import numpy as np
import random
import matplotlib.pyplot as plt
from kmeans import kmeans
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
# from my_attempt import Dataset 
from utils import simplex_opt, get_data, CLR_map, l2_dist, calc_eigen, get_laplacian
import evaluation

def CAN(data, num_category, lambda_1 = 1, epoch = 20):
    """
    
    Arguments
    ---------
    data : raw data 
    k : number of clusters
    lambda_c : parameter which controls the degree of laplacian rank constrained  
    epoch : max_iteration
    Returns
    -------
    S : graph constructed by CAN(n*n)
    G : clustering results (n dim vector)
    """
    
    
    num_data = data.shape[0]
    affinity = CLR_map(data, m = 4)
    laplacian = get_laplacian(affinity)
    eps = 1e-7
    lamb, Feature = calc_eigen(laplacian, num_category)        
    # print(lamb[0: 3*k])
    distance = np.zeros((num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            distance[i, j] = l2_dist(data[i], data[j])
    for t in range(epoch):
        d = np.zeros((num_data, num_data))
        for i in range(num_data):
            for j in range(num_data):
                d[i, j] = distance[i, j] + lambda_1 * l2_dist(Feature[i], Feature[j])
            affinity[i] = simplex_opt(-1.0/(2 * gamma) * d[i])
        laplacian = get_laplacian(affinity, normalization=0)
        F_old = Feature.copy()
        lamb, Feature = calc_eigen(laplacian, num_category)
        if np.sum(lamb[0:num_category]) > eps:
            lambda_1 = 2 * lambda_1
        else :
            if np.sum(lamb[0:num_category + 1]) < eps:
                lambda_1 = lambda_1 / 2
                Feature = F_old.copy()
            else :
                break 
    _, G = connected_components(affinity)
    if _ != num_category :
        print("Wrong Clustering", _)
    return affinity, G 
def get_gamma(data, m = 8):
    """
    
    Arguments
    ---------
    data : raw data 
    m : number of neighbors taken into consider
    Returns
    -------
    gamma : a hyperparameter in CAN
    """
    
    
    n = data.shape[0]
    e = np.zeros((n, n))    
    for i in range(n):
        for j in range(n):
            e[i, j] = l2_dist(data[i], data[j])
    idx = np.zeros((n, m + 1))
    for i in range(n):
        idx[i] = np.argsort(e[i])[:m + 1]
    idx = idx.astype(np.int16)
    # print("idx =", idx)
    W = np.zeros((n, n))
    rr = np.zeros(n)
    for i in range(n):
        id = idx[i, 1:m + 1]
        d = e[i, id]
        rr[i] = (m * d[m - 1] - np.sum(d))
    return np.mean(rr)
filename = ""
if __name__ == "__main__":
    Dataset = "Mfeat"
    Dataset = "Caltech101-7"
    data, label, num_category = get_data(dataset = Dataset)
    # gamma = 2
    lambda_1 = 1
    for view in range(len(data)):
        data_v = data[view].copy()
        gamma = get_gamma(data_v, m = 20)
        print(gamma)
        S, ans = CAN(data_v, num_category, lambda_1=lambda_1)
        for i in range(num_category):
            print("Cluster_num" + str(i) + ":", (ans == i).sum())
        
        f = open("acc1.txt", "a+")
        print("CLR:", evaluation.clustering(ans, label), file = f)    
        print(evaluation.clustering(ans, label))
        f.close()