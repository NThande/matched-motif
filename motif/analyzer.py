import numpy as np

import cluster
import config as cfg
import graphutils as graph
import segmentation as seg
import match_filter
import landmark_filter
import motifutils as motif

NODE_LABEL = cfg.NODE_LABEL
CLUSTER_NAME = cfg.CLUSTER_NAME
CLUSTER_EDGE_NAME = cfg.CLUSTER_EDGE_NAME


def analyze(audio, fs,
            num_motifs=cfg.N_ClUSTERS,
            seg_length=cfg.SEGMENT_LENGTH,
            threshold=0,
            seg_method='beat',
            cluster_method='agglom',
            similarity_method='match',
            with_topk=True,
            with_graph=True,
            with_overlap=False,
            with_reweight=True,
            with_fill=True,
            with_join=True):
    # Segmentation and Similarity Calculation
    _, _, segments, adjacency = self_similarity(audio, fs,
                                                method=similarity_method,
                                                length=seg_length,
                                                seg_method=seg_method)
    print("Done Self-Similarity")

    # Avoid overlap effects for any window
    if with_overlap is False:
        adjacency = remove_overlap(segments, adjacency, seg_length)

    # Add additional weights for segments that are close together in time
    if with_reweight:
        adjacency = reweight_by_time(segments, adjacency)

    # Create labels for nodes
    time_labels = seg.seg_to_label(segments, segments + seg_length)

    # Adjacency matrix thresholding
    if with_topk:
        adjacency = topk_threshold(adjacency, threshold)
    else:
        adjacency[adjacency < threshold] = 0

    # Incidence matrix clustering
    G = None
    if with_graph:
        # Create networkx graph and extract its incidence matrix for clustering
        G, relabel_idx = graph.adjacency_matrix_to_graph(adjacency, time_labels, NODE_LABEL, prune=True)
        motif_labels = cluster.cluster(incidence=None, graph=G, k_clusters=num_motifs, method=cluster_method)

        # Add clustering results to graph attributes
        graph.add_node_attribute(G, motif_labels, CLUSTER_NAME)
        graph.node_to_edge_attribute(G, CLUSTER_NAME, CLUSTER_EDGE_NAME, from_source=True)

    else:
        # Create incidence matrix and cluster directly
        M, _ = graph.adjacency_to_incidence_matrix(adjacency, prune=False)
        M, relabel_idx = graph.prune_incidence(M)
        motif_labels = cluster.cluster(incidence=M, k_clusters=num_motifs, method=cluster_method)

    # Update segment list to account for pruned nodes
    seg_starts = segments[relabel_idx]
    seg_ends = seg_starts + seg_length

    # Merge motifs and rebuild text labels
    seg_starts, seg_ends, motif_labels = motif.merge_motifs(seg_starts, seg_ends, motif_labels)
    motif_labels = motif.sequence_labels(motif_labels)

    if with_join:
        seg_starts, seg_ends, motif_labels = motif.motif_join(seg_starts, seg_ends, motif_labels)

        # Prune short motifs outs of full segmentation, fill any new gaps
        seg_starts, seg_ends, motif_labels = motif.prune_motifs(seg_starts, seg_ends, motif_labels,
                                                                min_length=cfg.SEGMENT_LENGTH / 2)

    if with_fill:
        seg_starts, seg_ends, motif_labels = motif.fill_motif_gaps(seg_starts, seg_ends, motif_labels,
                                                                   gap_length=cfg.SEGMENT_LENGTH / 2)
    motif_labels = motif.sequence_labels(motif_labels)
    return seg_starts, seg_ends, motif_labels, G


# Choose a method for calculating similarity
def self_similarity(audio, fs, length, method, seg_method):
    method = method.lower()
    if method == 'match':
        return match_filter.thumbnail(audio, fs, length=length, seg_method=seg_method)
    elif method == 'shazam':
        return landmark_filter.thumbnail(audio, fs, length=length, seg_method=seg_method)
    else:
        print("Unrecognized similarity method: {}".format(method))


# Performs a hard threshold on an adjacency matrix with different methods
def topk_threshold(adjacency, threshold):
    # Keep only the top k connections for each node
    print("Pre-Threshold: {num_edges}, threshold = {thresh}".format(num_edges=np.count_nonzero(adjacency),
                                                                    thresh=threshold))
    if threshold >= 1:
        num_nodes = adjacency.shape[0]
        k = int(threshold)
        if num_nodes < k:
            return adjacency
        for i in range(num_nodes):
            row = adjacency[i, :]
            kth_entry = row[np.argsort(row)[num_nodes - 1 - k - 1]]
            row[row < kth_entry] = 0
            adjacency[i, :] = row
    # Only keep top proportion of nodes
    elif threshold > 0:
        adj_vals = adjacency.flatten()
        adj_vals = np.sort(adj_vals[adj_vals.nonzero()])
        adj_vals_idx = int(adj_vals.shape[0] * threshold)
        thresh_val = adj_vals[adj_vals_idx]
        adjacency[adjacency < thresh_val] = 0
    else:
        return adjacency
    print("Post-Threshold: {num_edges}, threshold = {thresh}".format(num_edges=np.count_nonzero(adjacency),
                                                                    thresh=threshold))
    return adjacency


# Avoid overlap effects for any window
def remove_overlap(segments, adjacency, length):
    num_segments = segments.shape[0]
    for i in range(num_segments):
        for j in range(num_segments):
            this_seg = segments[i]
            that_seg = segments[j]
            if this_seg <= that_seg <= this_seg + length:
                adjacency[i, j] = 0
                adjacency[j, i] = 0
            elif that_seg <= this_seg <= that_seg + length:
                adjacency[i, j] = 0
                adjacency[j, i] = 0

    adjacency = seg.row_normalize(segments, adjacency)
    return adjacency


# Add additional weighting to adjacent segments
def reweight_by_time(segments, adjacency):
    # A perfect overlap is granted a score of 1. Total disparity (opposite ends of track) is a score of 0.
    num_segments = segments.shape[0]
    last_seg = segments[num_segments - 1]
    dist_matrix = np.zeros(adjacency.shape)
    for i in range(num_segments):
        cur_seg = segments[i]
        for j in range(num_segments):
            dist_matrix[i, j] = 1 - (np.abs(cur_seg - segments[j]) / last_seg)

    # Adjust weights and re-normalize
    adjacency = (0.1 * dist_matrix) + adjacency
    adjacency = seg.row_normalize(segments, adjacency)

    return adjacency


def main():
    # Create synthetic audio sequence
    return


if __name__ == '__main__':
    main()
