import networkx as nx
import fileutils
import matchfilt as mf
import numpy as np
from sklearn import cluster as skcluster
import visutils as vis
import graphutils as graph


# Choose a clustering method from a given input
def cluster(g, k_clusters, method='kmeans', **kwargs):
    clusters = None
    if method is 'kmeans':
        clusters = k_means(g, k_clusters, **kwargs)
    elif method is 'agglom':
        clusters = agglom(g, k_clusters, **kwargs)
    else:
        print("Unrecognized clustering method: {}".format(method))
    return clusters


# Cluster using k means with k_clusters clusters
def k_means(g, k_clusters, incidence=None, weight='weight', n_init=200, **kwargs):
    if incidence is None:
        incidence = nx.incidence_matrix(g, weight=weight)
    kmeans_clf = skcluster.KMeans(n_clusters=k_clusters, n_init=n_init)
    kmeans = kmeans_clf.fit_predict(incidence)
    return kmeans


# Cluster using agglomerative clustering, starting with k_clusters clusters.
def agglom(g, k_clusters=2, incidence=None, weight='weight', linkage='ward', **kwargs):
    if incidence is None:
        incidence = nx.incidence_matrix(g, weight=weight)
    agglom_clf = skcluster.AgglomerativeClustering(n_clusters=k_clusters, linkage=linkage)
    agglom = agglom_clf.fit_predict(incidence)
    return agglom


def main():
    return


if __name__ == '__main__':
    main()
