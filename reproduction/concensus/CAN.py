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


def plot_data(data, catagory):
    cr = []
    color = ["r", "b", "y", "g"]
    for i in range(data.shape[0]):
        cr.append(color[catagory[i]])
    plt.scatter(data[:, 0], data[:, 1], color=cr)
    plt.plot()
    plt.show()



def CAN(data, k, lambda_1 = 1, epoch = 20):
    N = data.shape[0]
    S = CLR_map(data, m = 4)
    L = get_laplacian(S)
    eps = 1e-7
    lamb, F = calc_eigen(L, k)        
    # print(lamb[0: 3*k])
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            A[i, j] = l2_dist(data[i], data[j])
        # A[i] = A[i] / np.sum(A[i])
            # print(data[i], data[j], A[i][j])
            # return
    # A /= np.sum(A)
    for t in range(epoch):
        d = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                d[i, j] = A[i, j] + lambda_1 * l2_dist(F[i], F[j])
            S[i] = simplex_opt(-1.0/(2 * gamma) * d[i])
        L = get_laplacian(S, normalization=0)
        F_old = F.copy()
        lamb, F = calc_eigen(L, k)
        print("epoch:", t, lambda_1, lamb[0:k].sum(), lamb[0:k + 1].sum())
        # print(lamb[0: 3*k])
        # print(lamb)
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
    return S, G 
def get_gamma(data, m = 8):
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
    data_v, label, k = get_data(dataset = Dataset)
    # gamma = 2
    lambda_1 = 1
    for ww in range(len(data_v)):
        data = data_v[ww].copy()
        gamma = get_gamma(data, m = 20)
        print(gamma)
        S, ans = CAN(data, k, lambda_1=lambda_1)
        for i in range(k):
            print("Cluster_num" + str(i) + ":", (ans == i).sum())
        
        f = open("acc1.txt", "a+")
        print("CLR:", evaluation.clustering(ans, label), file = f)    
        print(evaluation.clustering(ans, label))
        f.close()