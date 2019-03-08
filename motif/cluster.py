import networkx as nx
from sklearn import cluster as skc


# Choose a clustering method from a given input
def cluster(incidence, k_clusters, method='kmeans', graph=None, weight='weight', **kwargs):
    clusters = None
    if incidence is None:
        if graph is None:
            print("No input Data")
            return clusters
        incidence = nx.incidence_matrix(graph, weight=weight)

    if method is 'kmeans':
        clusters = k_means(incidence, k_clusters, **kwargs)
    elif method is 'agglom':
        clusters = agglom(incidence, k_clusters, **kwargs)
    else:
        print("Unrecognized clustering method: {}".format(method))
    return clusters


# Cluster using k means with k_clusters clusters
def k_means(incidence, k_clusters, n_init=200, **kwargs):
    kmeans_clf = skc.KMeans(n_clusters=k_clusters, n_init=n_init, **kwargs)
    kmeans = kmeans_clf.fit_predict(incidence)
    return kmeans


# Cluster using agglomerative clustering, starting with k_clusters clusters.
def agglom(incidence, k_clusters=2, linkage='ward', **kwargs):
    agglom_clf = skc.AgglomerativeClustering(n_clusters=k_clusters, linkage=linkage)
    agglom = agglom_clf.fit_predict(incidence)
    return agglom


def main():
    return


if __name__ == '__main__':
    main()
