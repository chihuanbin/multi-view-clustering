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
gamma = 10
Dataset = "100leaves"
# Dataset = "Caltech101-7"
data_v, label, k = get_data(dataset=Dataset)
def SwMC(W_v, k, lambda_1 = 1):
    epoch = 0
    V = len(W_v)
    alpha = np.ones(V) / V
    S = np.zeros((n, n))
    for v in range(V):
        S += alpha[v] * W_v[v]    
    W = S.copy()            
    L = get_laplacian(S, normalization=0)
    lamb, F = calc_eigen(L, k)
    last_loss = 0 
    while epoch < 20:  
        epoch += 1
        print("epoch:", epoch)
        S, G = CLR(W, k)
        cur_loss = 0
        for v in range(V):
            temp = l2_dist(S, W_v[v])
            alpha[v] = 1/(2*np.sqrt(temp))
            cur_loss += temp
        W = alpha[0] * W_v[0] 
        for v in range(1, V):
            W += alpha[v] * W_v[v] 
        W /= np.sum(alpha)
        _, G = connected_components(S)
        if _ != k :
            print("Wrong Clustering", _)
        print("SwMC:", evaluation.clustering(G, label))    
        for i in range(k):
            print("Cluster_num" + str(i) + ":", (G == i).sum())

        # F_old = F.copy()            
        # L = get_laplacian(S, normalization=0)
        # lamb, F = calc_eigen(L, k)
        
    _, G = connected_components(S)
    if _ != k :
        print("Wrong Clustering", _)
    print("SwMC:", evaluation.clustering(G, label))    
    for i in range(k):
        print("Cluster_num" + str(i) + ":", (G == i).sum())
    return _, G
def PwMC(W_v, k, lambda_c = 1):
    V = len(W_v)
    n = W_v[0].shape[0]
    alpha = np.ones(V) / V
    W = np.zeros((n, n))
    for v in range(V):
        W += alpha[v] * W_v[v]
    L = get_laplacian(W, normalization=0)
    lamb, F = calc_eigen(L, k)
    S = W.copy()
    for epoch_out in range(5):
        for epoch_in in range(30):
            c = dist_map(F)
            for i in range(n):
                vec = W[i] - lambda_c / 2 * c[i]
                vec /= np.sum(alpha)
                S[i] = simplex_opt(vec)
            F_old = F.copy()
            L = get_laplacian(S, normalization=0)
            lamb, F = calc_eigen(L, k)

            print("epoch:", epoch_out, lambda_c, lamb[0:k].sum(), lamb[0:k + 1].sum())
            if np.sum(lamb[0:k]) > eps:
                lambda_c = 2 * lambda_c
            else :
                if np.sum(lamb[0:k + 1]) < eps:
                    lambda_c = lambda_c / 2
                    F = F_old.copy()
                else :
                    break
        e = np.zeros(V)
        for v in range(V):
            e[v] = l2_dist(S, W_v[v])

        alpha = simplex_opt(-0.5 * e/ gamma) 
        W = alpha[0] * W_v[0] 
        for v in range(1, V):
            W += alpha[v] * W_v[v] 
        print(alpha)
        _, G = connected_components(S)
        if _ != k :
            print("Wrong Clustering", _)
        print("PwMC:", evaluation.clustering(G, label))    
        for i in range(k):
            print("Cluster_num" + str(i) + ":", (G == i).sum())
    return _, G
n = label.shape[0]
W_v = []
V = len(data_v)
for i in range(V):
    W = CLR_map(data_v[i])
    W_v.append(W)
lambda_c = 1
_, _ = PwMC(W_v, k)
