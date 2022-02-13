import os
from sklearn import metrics
import numpy as np
import random
import matplotlib.pyplot as plt


import sys
sys.path.append("./")

from kmeans import kmeans
from utils import simplex_opt, get_data, CLR_map, l2_dist, acc, calc_eigen,get_laplacian
from sklearn.metrics.cluster import normalized_mutual_info_score
import evaluation

eps = 1e-7
# seeds = [1, 10, 100]
# for seed in seeds:
seed = 10

# def CLR_map(data, m = 4):
#     n = data.shape[0]
#     e = np.zeros((n, n))    
#     for i in range(n):
#         for j in range(n):
#             e[i, j] = l2_dist(data[i], data[j])
#     # print("calc_dis finished")
#     idx = np.zeros((n, m + 1))
#     for i in range(n):
#         idx[i] = np.argsort(e[i])[:m + 1]
        
#     idx = idx.astype(np.int16)
#     W = np.zeros((n, n))
#     eps = 1e-8
#     for i in range(n):
#         id = idx[i, 1:m + 1]
#         d = e[i, id]
#         W[i, id] = (d[m - 1] - d + eps / m) / (m * d[m - 1] - np.sum(d) + eps)
#     return (W + W.T) / 2

def build_net(data):
    V = len(data)
    S = []
    for v in range(V):
        data_v = data[v]
        S.append(CLR_map(data_v))
    return S
# if __name__ == "main":
# Dataset = "Mfeat"
Dataset = "Caltech101-7"
# Dataset = "100leaves"
# Dataset = "bbcsport"
data, label, k = get_data(dataset = Dataset)
V = len(data)
print("111111111")


N = label.shape[0]



counter = 0
F_c = np.zeros((N, k))
best_ans = 0
W = []

'''
W: n*m 
F: m*k

'''
S = build_net(data)
# for v in range(V):
#     print(np.where(np.sum(S[v]) < eps))
L = []
for v in range(V):
    d = np.sum(S[v], axis=0)
    D = np.diag(d)    
    D_w = np.diag(np.sqrt(1/d)) + eps
    L.append(D_w @ (D - S[v]) @ D_w)  
    # L.append(D - S[v])

converge = 0
last_lambda = 0

counter = 0
F_c = np.zeros((N, k))
lambda_w = 0.01
while True:
    F = []  
    for v in range(V):
        W_v = S[v]
        complete = L[v] - lambda_w * (F_c@F_c.T) #均值不为0;;
        _, F_v = calc_eigen(complete, k)
        F.append(F_v)

    reg_avg = np.zeros((N,N))
    for v in range(V):
        reg_avg += F[v]@F[v].T
    L_c = get_laplacian(reg_avg)

    _, F_c = calc_eigen(L_c, k, 0)    
    counter += 1
    _, ans = kmeans(F_c, k)
    for i in range(k):
        print("Cluster_num" + str(i) + ":", (ans == i).sum())
    if counter > 10 :
        break

    _, ans = kmeans(F_c, k)
    # f = open("ans.txt", "a+")
    result = evaluation.clustering(ans, label)["kmeans"]
    print(counter, result["accuracy"] )
    if result["accuracy"] > best_ans :
        best_ans = result["accuracy"]
        best_result = result


print(best_result)