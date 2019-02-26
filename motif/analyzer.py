import fileutils
import matchfilt as mf
import pairfilt as pf
import numpy as np
import visutils as vis
import graphutils as graph
import cluster
import config as cfg

NODE_LABEL = cfg.NODE_LABEL
CLUSTER_NAME = cfg.CLUSTER_NAME
CLUSTER_EDGE_NAME = cfg.CLUSTER_EDGE_NAME


def analyze(audio, fs,
            num_motifs=3,
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

    # Graph Creation
    adjacency[adjacency < threshold] = 0
    G = graph.adjacency_matrix_to_graph(adjacency, labels, NODE_LABEL, prune=True)

    # Clustering method
    motif_labels = cluster.cluster(G, num_motifs, method=cluster_method, **kwargs)
    unique_motifs = np.unique(motif_labels)
    motifs = dict.fromkeys(unique_motifs)
    for i in G.nodes:
        if motifs[motif_labels[i]] is None:
            motifs[motif_labels[i]] = []
        else:
            motifs[motif_labels[i]].append(G.nodes[i][NODE_LABEL])

    graph.add_node_attribute(G, motif_labels, CLUSTER_NAME)
    graph.node_to_edge_attribute(G, CLUSTER_NAME, CLUSTER_EDGE_NAME, from_source=True)

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
    return


if __name__ == '__main__':
    main()
