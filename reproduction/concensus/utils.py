import os
from sklearn import metrics
import numpy as np
import random
import matplotlib.pyplot as plt
from kmeans import kmeans
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components        
        
        
def CLR_map(data, m = 5):
    n = data.shape[0]
    e = np.zeros((n, n))    
    for i in range(n):
        for j in range(n):
            e[i, j] = l2_dist(data[i], data[j])
    # print("calc_dis finished")
    idx = np.zeros((n, m + 1))
    for i in range(n):
        idx[i] = np.argsort(e[i])[:m + 1]
    idx = idx.astype(np.int16)
    # print("idx =", idx)
    W = np.zeros((n, n))
    eps = 1e-8
    for i in range(n):
        id = idx[i, 1:m + 1]
        d = e[i, id]
        W[i, id] = (d[m - 1] - d) / (m * d[m - 1] - np.sum(d) + eps)
    return W

def simplex_opt(v, k=1):
    '''
    Column Vector :param v:
    IDK parameter :param k:
    :return:
    '''
    n = v.shape[0]
    u = v - v.mean() + 1 / n

    if np.min(u) < 0:
        f = 1
        turn = 0
        lambda_b = 0
        while (abs(f) > 1e-10):
            turn += 1
            u_1 = u - lambda_b
            p_idx = (u_1 > 0)
            # f = np.sum(u_1[p_idx]) - n * lambda_b
            q_idx = (u_1 < 0)
            f = np.sum(np.maximum(-u_1[q_idx], 0)) - n * lambda_b
            g = np.sum(q_idx) - n
            # f = np.sum(u_1[p_idx]) - k
            # g = -np.sum(p_idx)
            lambda_b = lambda_b - f / g
            if turn > 100:
                print("Diverge!!!!")
                break
        x = np.maximum(u_1, 0)
    else:
        x = u
    return x
#"Mfeat"
def get_data(dataset="Caltech101-7"):
    dir = "C:/Users/78258/Desktop/repos/Reproduction/data_process/processed/"
    # dir = "./data_process/processed/"
    data_v = []
    for filename in os.listdir(dir):
        if filename.split(".")[0] == dataset:
            datas = np.load(dir + filename, allow_pickle=True)
            data = datas.item()["X"].squeeze()
            label = datas.item()["Y"].squeeze()
            for i in range(data.shape[0]) :
                data1 = data[i].astype(float)
                # print(data1.shape)
                for j in range(data1.shape[0]):
                    data1[j, :] = (data1[j, :] - data1[j, :].mean()) /( data1[j, :].std() + 1e-3)
                # print(data1)
                data_v.append(data1.T)
            # print(label[0].squeeze())
            for i in range(1, label.shape[0]) :
                if label[i].any().squeeze() != label[i - 1].any().squeeze():
                    print("wrong_label")
            if label.shape[0] < 50:
                label = label[0].squeeze()
            # print(type(label), label.shape, label)
            label = label.astype(np.int)
            label -= np.min(label)
            k = label.max() + 1
            print("Data Gotten")
            return data_v, label, k
    print("Not Found")
    return 
def get_laplacian(W, normalization = 1):
    '''
    W: Adjacency matrix(n*n)
    normalization: N or NotN
    output:
    L: Laplace Matrix
    '''
    S = (W + W.T) / 2
    d = np.sum(S, axis=0)
    D = np.diag(d)
    L = D - S
    if normalization == 1:
        D_w = np.diag(np.sqrt(1/d))
        L = D_w @ L @ D_w
    return L 

def calc_eigen(L, k, large_ones = 0):
    '''
    L: Laplace Matrix
    k: Clustering number
    output:
    lamb: all_lambda (sorted)
    F: Eigenvector
    '''

    lamb, V = np.linalg.eig(L) #不稳定？

    lamb = lamb.real
    if large_ones == 0:
        idx = np.argsort(lamb)[:k]
    else :
        idx = np.argsort(lamb)[::-1][:k] 
    lamb = np.sort(lamb)
    F = V[:, idx]
    for i in range(F.shape[1]) :
        F[:, i] = F[:, i] / np.sqrt(np.sum(F[:, i] ** 2))
    return lamb, F.real

def l2_dist(x, y):
    return np.sum((x - y) ** 2)
def f_norm(x, y):
    return np.sqrt(l2_dist(x, y))
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
