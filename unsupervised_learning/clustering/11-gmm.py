#!/usr/bin/env python3
'''clustering'''
import sklearn.mixture


def kmeans(X, k):
    '''
    - X is a numpy.ndarray of shape (n, d)
    containing the dataset
    - k is the number of clusters
    '''
    gmm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    s = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
