import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans
from utils import get_data
from sklearn import cluster
import sklearn
# print(sklearn.__version__)
def eu_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def l2_dist(x, y):
    return np.sum((x - y) ** 2)

def affinity(data, method = "k-nearest", k = 5, sigma = 10):
    n = data.shape[0]
    dist = np.zeros(shape = (n, n))
    a = np.zeros(shape = (n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = l2_dist(data[i], data[j])
        w = np.argsort(dist[i])[1:k+1]   
        for j in w:         
            a[i][j] = np.exp(-dist[i][j] /(2*sigma*sigma))
    return (a + a.T) / 2

def CLR_map(data, m = 4):
    n = data.shape[0]
    e = np.zeros((n, n))    
    for i in range(n):
        for j in range(n):
            e[i, j] = l2_dist(data[i], data[j])
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
    return (W + W.T) / 2    
def plot_data(data, catagory, name = None) :
    plt.figure()
    cr = []
    color = ["r", "b","y", "g"]
    for i in range(data.shape[0]):
        cr.append(color[catagory[i]])
    plt.scatter(data[:, 0], data[:, 1], color = cr)
    plt.plot()
    if name != None:
        plt.savefig(name)
    plt.close()
    # plt.show()

def spectral_clustering(W, k):
    """
    
    Arguments
    ---------
    W : adjacency matrix of data
    k : number of clusters
    
    Returns
    -------
    catagory : clustering results (n dim vector)
    C : low-dimensional feature obtained by spectral clustering
    """
    
    
    n = W.shape[0]
    d = np.sum(W, axis = 0)
    D = np.diag(d)
    # L = D - W
    # print(L)
    L = np.dot(np.dot(np.diag(1/np.sqrt(d)), (D - W)), np.diag(1/np.sqrt(d))) 
    # print(L)
    lamb, V = np.linalg.eig(L)
    lamb = lamb.real
    idx = np.argsort(lamb)
    C = np.zeros((n, k))
    C = V[:, idx[0:k]]
    for i in range(C.shape[1]):
        C[:, i] = C[:, i] / np.sqrt(np.sum(C[:, i] ** 2))
    # plot_data(C, np.ones(n, dtype=int))
    _, catagory = kmeans(C, k) 
    # plot_data(C, catagory)
    return catagory, C
n = 15
k = 3
# data = get_data(n, k)
Dataset = "Mfeat"
Dataset = "Caltech101-7"
data, label, k = get_data(dataset = Dataset)
for v in range(len(data)):
    data1 = data[v]
    n = label.shape[0]
    W = CLR_map(data1)
    catagory, C = spectral_clustering(W, k)
    import evaluation
    print("ANS:", evaluation.clustering(catagory, label))

# catagory = cluster.spectral_clustering(affinity = W, n_clusters = 2)

# for i in np.linspace(0.10, 0.20, 20):
#     # W = affinity(data, sigma = i)
#     plot_data(C, catagory, "C:/Users/78258/Desktop/codes/reproduction/Pics/spectral_"+str(i)+".png")
#     plot_data(data, catagory, "C:/Users/78258/Desktop/codes/reproduction/Pics/sigma_"+str(i)+".png")