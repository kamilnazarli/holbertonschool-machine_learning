#!/usr/bin/env python3
'''clustering'''
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    '''
    - X is a numpy.ndarray of shape
    (n, d) containing the dataset
    - dist is the maximum cophenetic
    distance for all clusters
    '''
    hierarchy = scipy.cluster.hierarchy
    linkage = hierarchy.linkage(X, method='ward')
    fcluster = hierarchy.fcluster(linkage, dist, criterion='distance')
    hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.figure()
    plt.show()
    return fcluster
