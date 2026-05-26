#!/usr/bin/env python3
'''clustering'''
import sklearn.cluster


def kmeans(X, k):
    '''
    - X is a numpy.ndarray of shape (n, d)
    containing the dataset
    - k is the number of clusters
    '''
    km = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return km.cluster_centers_, km.labels_
