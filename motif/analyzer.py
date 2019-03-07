import numpy as np

import cluster
import config as cfg
import graphutils as graph
import matchfilt as mf

NODE_LABEL = cfg.NODE_LABEL
CLUSTER_NAME = cfg.CLUSTER_NAME
CLUSTER_EDGE_NAME = cfg.CLUSTER_EDGE_NAME


def analyze(audio, fs,
            num_motifs,
            seg_length=cfg.SEGMENT_LENGTH,
            threshold=50.,
            tf_method='stft',
            seg_method='onset',
            cluster_method='kmeans',
            similarity_method='match',
            with_graph=True,
            **kwargs):
    # Segmentation and Similarity Calculation
    _, _, segments, adjacency = thumbnail(audio, fs,
                                          method=similarity_method,
                                          length=seg_length,
                                          seg_method=seg_method,
                                          tf_method=tf_method,
                                          **kwargs)

    # Create labels for nodes
    labels = []
    for i in range(segments.shape[0]):
        labels.append('{start:2.2f} - {end:2.2f}'.format(start=segments[i], end=segments[i] + seg_length))

    # Threshold Adjacency Matrix
    adjacency[adjacency < threshold] = 0

    # Incidence matrix clustering
    G = None
    relabels = []

    # Create networkx graph and cluster with its incidence matrix
    if with_graph:
        G = graph.adjacency_matrix_to_graph(adjacency, labels, NODE_LABEL, prune=True)
        motif_labels = cluster.cluster(incidence=None, graph=G, k_clusters=num_motifs, method=cluster_method, **kwargs)

        # Add clustering results to graph attributes
        graph.add_node_attribute(G, motif_labels, CLUSTER_NAME)
        graph.node_to_edge_attribute(G, CLUSTER_NAME, CLUSTER_EDGE_NAME, from_source=True)

        # Create new label list for motif grouping
        for i in G.nodes:
            relabels.append(G.nodes[i][NODE_LABEL])

    # Create incidence matrix and cluster directly
    else:
        M, relabels = graph.adjacency_to_incidence_matrix(adjacency, labels, prune=True)
        motif_labels = cluster.cluster(incidence=M, k_clusters=num_motifs, method=cluster_method, **kwargs)

    # Collect motifs into a single output dictionary
    motifs = dict.fromkeys(np.unique(motif_labels))
    for i in range(len(relabels)):
        if motifs[motif_labels[i]] is None:
            motifs[motif_labels[i]] = []
        else:
            motifs[motif_labels[i]].append(relabels[i])

    return motifs, G


# Choose a method for calculating similarity
def thumbnail(audio, fs, length, method='match', seg_method='onset', **kwargs):
    if method is 'match':
        return mf.thumbnail(audio, fs, length=length, seg_method=seg_method, **kwargs)
    # elif method is 'pair':
    #     return pf.thumbnail(audio, fs, length=length, seg_method=seg_method)
    else:
        print("Unrecognized clustering method: {}".format(method))


def main():
    # Create synthetic audio sequence
    return


if __name__ == '__main__':
    main()
