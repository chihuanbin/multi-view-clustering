import numpy as np
import random
import matplotlib.pyplot as plt


def eu_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def l2_dist(x, y):
    return np.sum((x - y) ** 2)

def calc_dist(x, y, method = "l2"):
    if method == "eu" :
        return eu_dist(x, y)
    if method == "l2" :
        return l2_dist(x, y)

def kmeans(data, k, iter_num = 400):
    """
    
    Arguments
    ---------
    data : raw data of size n*n_f
    k : num of clusters
    iter_num : maximum number of iterations
    Returns
    -------
    centroid : centroid derived from k-means
    catagory : clustering results
    """
    
    
    eps = 1e-6
    n = data.shape[0]
    n_f = data.shape[1]
    centroid = np.zeros((k, n_f), dtype = float)                                 #centroid of k th clustering
    catagory = np.random.randint(low = 0, high = k, size = n, dtype = int)           #data point belong to which catagory
    dis_list = []
    for i in range(k):
        k_num = np.sum(catagory == i) 
        for j in range(n_f) :     
            if k_num == 0:
                centroid[i, j] = 0
            else :
                centroid[i, j] = np.sum(data[:, j] * (catagory == i)) / (np.sum(catagory == i))
    for w in range(iter_num):
 
        all_dist = 0
        for i in range(n):
            dist = np.zeros(shape=k)
            for j in range(k):
                dist[j] = calc_dist(data[i], centroid[j])
            catagory[i] = np.argmin(dist)
            all_dist += dist[catagory[i]] 
        all_dist = 0
        for i in range(k):
            k_num = np.sum(catagory == i) 
            for j in range(n_f) :     
                if k_num == 0:
                    centroid[i, j] = 0
                else :
                    centroid[i, j] = np.sum(data[:, j] * (catagory == i)) / (np.sum(catagory == i))

        for i in range(n):
            for j in range(k):
                if catagory[i] == j:
                   all_dist +=  calc_dist(data[i], centroid[j])
        if len(dis_list) > 0 and dis_list[len(dis_list) - 1] - all_dist < eps:
            break
        dis_list.append(all_dist)
    return centroid, catagory        
