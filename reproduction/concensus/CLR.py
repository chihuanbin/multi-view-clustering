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


def plot_data(data, catagory):
    cr = []
    color = ["r", "b", "y", "g"]
    for i in range(data.shape[0]):
        cr.append(color[catagory[i]])
    plt.scatter(data[:, 0], data[:, 1], color=cr)
    plt.plot()
    plt.show()


# def simplex_opt(v, k=1):
#     '''
#     Column Vector :param v:
#     IDK parameter :param k:
#     :return:
#     '''
#     n = v.shape[0]
#     u = v - v.mean() + 1 / n

#     if np.min(u) < 0:
#         f = 1
#         turn = 0
#         lambda_b = 0
#         while (abs(f) > 1e-10):
#             turn += 1
#             u_1 = u - lambda_b
#             p_idx = (u_1 > 0)
#             # f = np.sum(u_1[p_idx]) - n * lambda_b
#             q_idx = (u_1 < 0)
#             f = np.sum(np.maximum(-u_1[q_idx], 0)) - n * lambda_b
#             g = np.sum(q_idx) - n
#             # f = np.sum(u_1[p_idx]) - k
#             # g = -np.sum(p_idx)
#             lambda_b = lambda_b - f / g
#             if turn > 100:
#                 print("Diverge!!!!")
#                 break

#         x = np.maximum(u_1, 0)
#     else:
#         x = u
#     return x


# def calc_eigen(W, k):
#     S = (W + W.T) / 2
#     d = np.sum(S, axis=0)
#     D = np.diag(d)
#     L = D - S
#     lamb, V = np.linalg.eig(L)
#     lamb = lamb.real
#     idx = np.argsort(lamb)[: k]
#     lamb = np.sort(lamb)
#     F = V[:, idx]
#     return lamb, F

def calc_uncertainty(S, A) :
    n = A.shape[0]
    u = np.zeros(n)
    for i in range(n):
        u[i] = l2_dist(S[i, :], A[i, :]) / l2_dist(S[i, :], 0)
    return u

def calc_probability(A, y):
    '''
    A: Ori graph
    y: Clustering result
    output
    b(n * k):
        probability of each sample
    '''
    eps = 1e-10
    n = A.shape[0]
    k = int(np.max(y) + 1)
    b = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
           b[i, j] = np.sum(A[i] * (y == j))
        b[i, :] = b[i, :] / np.sum(b[i, :] + eps)
    return b

def fusion(b1, u1, b2, u2) :
    '''
    b1, b2 (n * k): clustering result
    u1, u2 (n * 1): uncertainty
    output:
    b, u: afterfusion
    '''
    eps = 1e-5
    u = np.zeros(n)
    b = np.zeros((n, k))
    for i in range(n) :
        for j in range(k):
            b[i, j] = b1[i, j] * b2[i, j] + b1[i, j] * u2[i] + b2[i, j] * u1[i]
            u[i] = u1[i] * u2[i]
        C = np.sum(b[i]) + u[i] + eps
        b[i, :] = b[i, :] / C
        u[i] = u[i] / C
    return b, u



def CLR(W, k, lambda_c=1, iter_num=40):
    eps = 1e-8
    n = W.shape[0]
    W = (W + W.T)/2
    # if S is None:
    #     S = W.copy()
    S = np.zeros((n,n))
    L = get_laplacian(W, normalization=0)
    lamb, F = calc_eigen(L, k)
    epoch = 0
    while epoch < iter_num:
        # print("EPOCH,", epoch, lambda_c)
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
        # if np.sum(np.abs(lamb[0:k + 1] - former_l[0:k + 1])) < 1e-3:
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


# def get_data(database="Mfeat"):
#     dir = "./data_process/processed/"
#     data_v = []
#     for filename in os.listdir(dir):
#         if filename.split(".")[0] == database:
#             datas = np.load(dir + filename, allow_pickle=True)
#             # print(datas)
#             data = datas.item()["X"].squeeze()
#             label = datas.item()["Y"].squeeze()
#             for i in range(data.shape[0]) :
#                 data1 = data[i]
#                 print(data1.shape)
#                 for j in range(data1.shape[0]):
#                     data1[i, :] = (data1[i, :] - data1[i, :].mean()) / data1[i, :].std()
#                 data_v.append(data1.T)
#             # print(label[0].squeeze())
#             for i in range(1, label.shape[0]) :
#                 if label[i].any().squeeze() != label[i - 1].any().squeeze():
#                     print("wrong_label")
#             label = label[0].squeeze().astype(np.int)
#             label -= np.min(label)
#             k = label.max()
#             print("Data Gotten")
#             return data_v, label, k
#     print("Not Found")
#     return 
#     # return data, label, k
filename = ""
if __name__ == "__main__":
    data_v, label, k = get_data(dataset = "Caltech101-7")
    # print(data_v[0].shape, label.shape)
    # print(k)
    # u = None
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
        # itself = np.linspace(0, n - 1, n, dtype=int)

        for i in range(n):
            idx[i] = np.argsort(e[i])[:m + 1]

        idx = idx.astype(np.int16)
        # print("idx =", idx)
        W = np.zeros((n, n))
        eps = 1e-8
        for i in range(n):
            id = idx[i, 1:m + 1]
            d = e[i, id]
            W[i, id] = (d[m - 1] - d + eps / m) / (m * d[m - 1] - np.sum(d) + eps)

        S, ans = CLR(W, k)

        for i in range(k):
            print("Cluster_num" + str(i) + ":", (ans == i).sum())

        # save_info = {"S" : S, "cata": ans, "W" : W}
        # np.save("./CLR/info_" + filename, save_info, allow_pickle=True)
        f = open("acc1.txt", "a+")
        print("CLR:", evaluation.clustering(ans, label), file = f)    
        print(evaluation.clustering(ans, label))
        f.close()
        # u1 = calc_uncertainty(S, W)
        # u1 = np.zeros(n)
        # b1 = calc_probability(W, ans)

        # if u is None:
        #     u = u1
        #     b = b1
        # else :
        #     b, u = fusion(b1, u1, b, u)
        # b = b1    
        # predict = np.zeros(n)
        # for i in range(n):
        #     predict[i] = np.argmax(b[i])
        
        # f = open("acc1.txt", "a+")
        # print(filename, acc(predict, label), np.sum(u), file=f)

    # predict = np.zeros(n)
    # for i in range(n):
    #     predict[i] = np.argmax(b[i])

    # print("acc:", acc(predict, label))
    # f = open("acc.txt", "a+")
    # print(filename, acc(predict, label), np.sum(u), file=f)
    # f.close()