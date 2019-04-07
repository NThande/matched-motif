import numpy as np

import cluster
import config as cfg
import graphutils as graph
import matchedfilter as mf
import segmentation as seg
import matplotlib.pyplot as plt
import librosa.display as lb
import visutils as vis
import pandas as pd

NODE_LABEL = cfg.NODE_LABEL
CLUSTER_NAME = cfg.CLUSTER_NAME
CLUSTER_EDGE_NAME = cfg.CLUSTER_EDGE_NAME


def analyze(audio, fs,
            num_motifs,
            seg_length=cfg.SEGMENT_LENGTH,
            # threshold=50.,
            threshold=1,
            tf_method='stft',
            seg_method='onset',
            cluster_method='kmeans',
            similarity_method='match',
            topk_thresh=True,
            with_graph=True,
            **kwargs):
    # Segmentation and Similarity Calculation
    _, _, segments, adjacency = self_similarity(audio, fs,
                                                method=similarity_method,
                                                length=seg_length,
                                                seg_method=seg_method,
                                                tf_method=tf_method,
                                                **kwargs)
    # Create labels for nodes
    time_labels = seg.seg_to_label(segments, segments + seg_length)

    # Adjacency matrix thresholding
    if topk_thresh:
        adjacency = topk_threshold(adjacency, threshold)
    else:
        adjacency[adjacency < threshold] = 0

    # Incidence matrix clustering
    G = None
    if with_graph:
        # Create networkx graph and extract its incidence matrix for clustering
        G, relabel_idx = graph.adjacency_matrix_to_graph(adjacency, time_labels, NODE_LABEL, prune=True)
        motif_labels = cluster.cluster(incidence=None, graph=G, k_clusters=num_motifs, method=cluster_method, **kwargs)

        # Add clustering results to graph attributes
        graph.add_node_attribute(G, motif_labels, CLUSTER_NAME)
        graph.node_to_edge_attribute(G, CLUSTER_NAME, CLUSTER_EDGE_NAME, from_source=False)

    else:
        # Create incidence matrix and cluster directly
        M, _ = graph.adjacency_to_incidence_matrix(adjacency, prune=False)
        M, relabel_idx = graph.prune_incidence(M)
        motif_labels = cluster.cluster(incidence=M, k_clusters=num_motifs, method=cluster_method, **kwargs)

    # Update segment list to account for pruned nodes
    seg_starts = segments[relabel_idx]
    seg_ends = seg_starts + seg_length

    # Merge motifs and rebuild text labels
    seg_starts, seg_ends, motif_labels = seg.merge_motifs(seg_starts, seg_ends, motif_labels)
    motif_labels = motif_labels.astype(int)
    motif_labels = seg.sequence_labels(motif_labels)
    return seg_starts, seg_ends, motif_labels, G


# Choose a method for calculating similarity
def self_similarity(audio, fs, length, method='match', seg_method='onset', **kwargs):
    if method is 'match':
        return mf.thumbnail(audio, fs, length=length, seg_method=seg_method, **kwargs)
    # elif method is 'pair':
    #     return pf.thumbnail(audio, fs, length=length, seg_method=seg_method)
    else:
        print("Unrecognized clustering method: {}".format(method))


# Performs a hard threshold on an adjacency matrix with different methods
def topk_threshold(adjacency, threshold):
    # Keep only the top k connections for each node
    if threshold >= 1:
        if threshold < 1:
            return np.zeros(adjacency.shape)
        k = int(threshold)
        for i in range(adjacency.shape[0]):
            row = adjacency[i, :]
            if row.shape[0] < k:
                break
            else:
                kth_entry = row[np.argsort(row)[k - 1]]
                row[row < kth_entry] = 0
                adjacency[i, :] = row
    # Only keep top proportion of nodes
    elif threshold >= 0:
        adj_vals = adjacency.flatten()
        adj_vals = np.sort(adj_vals[adj_vals.nonzero()])
        adj_vals_idx = int(adj_vals.shape[0] * (1 - threshold))
        thresh_val = adj_vals[adj_vals_idx]
        adjacency[adjacency < thresh_val] = 0
    else:
        return adjacency
    return adjacency


def main():
    # Create synthetic audio sequence
    return


if __name__ == '__main__':
    main()
