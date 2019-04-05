import numpy as np

import cluster
import config as cfg
import graphutils as graph
import matchfilt as mf
import matplotlib.pyplot as plt
import librosa.display as lb

NODE_LABEL = cfg.NODE_LABEL
CLUSTER_NAME = cfg.CLUSTER_NAME
CLUSTER_EDGE_NAME = cfg.CLUSTER_EDGE_NAME


def analyze(audio, fs,
            num_motifs,
            seg_length=cfg.SEGMENT_LENGTH,
            # threshold=50.,
            threshold=50,
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
    # time_labels = []
    # for i in range(segments.shape[0]):
    #     time_labels.append('{start:2.2f} - {end:2.2f}'.format(start=segments[i], end=segments[i] + seg_length))
    time_labels = seg_to_label(segments, segments + seg_length)

    adjacency = hard_threshold(adjacency, threshold)
    # adjacency[adjacency < threshold] = 0

    # Incidence matrix clustering
    G = None
    # time_labels = []

    # Create networkx graph and cluster with its incidence matrix
    if with_graph:
        G, relabel_idx = graph.adjacency_matrix_to_graph(adjacency, time_labels, NODE_LABEL, prune=True)
        # seg_starts = segments[relabel_idx]
        motif_labels = cluster.cluster(incidence=None, graph=G, k_clusters=num_motifs, method=cluster_method, **kwargs)

        # Add clustering results to graph attributes
        graph.add_node_attribute(G, motif_labels, CLUSTER_NAME)
        graph.node_to_edge_attribute(G, CLUSTER_NAME, CLUSTER_EDGE_NAME, from_source=False)

        # Create new label list for motif grouping
        # time_labels = np.array(time_labels)[relabel_idx]

        # for i in G.nodes:
        #     relabels.append(G.nodes[i][NODE_LABEL])

    # Create incidence matrix and cluster directly
    else:
        M, _ = graph.adjacency_to_incidence_matrix(adjacency, prune=False)
        M, relabel_idx = graph.prune_incidence(M)
        motif_labels = cluster.cluster(incidence=M, k_clusters=num_motifs, method=cluster_method, **kwargs)

    # Create new label list to match motif grouping
    num_motifs = np.unique(motif_labels).shape[0]
    time_labels = np.array(time_labels)[relabel_idx]
    seg_starts = segments[relabel_idx]
    seg_ends = seg_starts + seg_length

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lb.waveplot(audio, fs, ax=ax, color='b')
    ax.set_xlim(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1)
    unique_motifs = {}
    for i in range(0, seg_starts.shape[0]):
        this_label = int(motif_labels[i])
        if this_label in unique_motifs:
            ax.axvspan(seg_starts[i], seg_ends[i], alpha=0.8, color='C{}'.format(this_label),
                       linestyle='-.')
        else:
            unique_motifs[this_label] = 1
            ax.axvspan(seg_starts[i], seg_ends[i], alpha=0.8, color='C{}'.format(this_label),
                       linestyle='-.', label = 'Event {}'.format(this_label))
    plt.title("Motif Segmentation found Through Matched Filter Before Merge")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Amplitude (dB)")
    ax.legend()

    seg_starts, seg_ends, motif_labels = merge_motifs(seg_starts, seg_ends, motif_labels)
    time_labels = seg_to_label(seg_starts, seg_ends)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lb.waveplot(audio, fs, ax=ax, color='b')
    ax.set_xlim(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1)
    unique_motifs = {}
    for i in range(0, seg_starts.shape[0]):
        this_label = int(motif_labels[i])
        if this_label in unique_motifs:
            ax.axvspan(seg_starts[i], seg_ends[i], alpha=0.8, color='C{}'.format(int(motif_labels[i])),
                       linestyle='-.')
        else:
            unique_motifs[this_label] = 1
            ax.axvspan(seg_starts[i], seg_ends[i], alpha=0.8, color='C{}'.format(this_label),
                       linestyle='-.', label = 'Event {}'.format(this_label))
    plt.title("Motif Segmentation found Through Matched Filter After Merge")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Amplitude (dB)")
    ax.legend()

    # Collect motifs into a single output dictionary
    motifs = dict.fromkeys(np.unique(motif_labels))
    for i in range(len(time_labels)):
        if motifs[motif_labels[i]] is None:
            motifs[motif_labels[i]] = [time_labels[i]]
        else:
            motifs[motif_labels[i]].append(time_labels[i])

    print(motifs)
    return motifs, G


# Choose a method for calculating similarity
def thumbnail(audio, fs, length, method='match', seg_method='onset', **kwargs):
    if method is 'match':
        return mf.thumbnail(audio, fs, length=length, seg_method=seg_method, **kwargs)
    # elif method is 'pair':
    #     return pf.thumbnail(audio, fs, length=length, seg_method=seg_method)
    else:
        print("Unrecognized clustering method: {}".format(method))


# Performs a hard threshold on an adjacency matrix with different methods
def hard_threshold(adjacency, threshold):
    # Keep only the top k connection for each node
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
    # Keep only the top proportion of connections
    elif threshold >= 0:
        adj_vals = adjacency.flatten()
        adj_vals = np.sort(adj_vals[adj_vals.nonzero()])
        adj_vals_idx = int(adj_vals.shape[0] * (1 - threshold))
        thresh_val = adj_vals[adj_vals_idx]
        adjacency[adjacency < thresh_val] = 0
    else:
        return adjacency
    return adjacency


# Create labels for nodes
def seg_to_label(starts, ends):
    labels = []
    for i in range(starts.shape[0]):
        labels.append('{start:2.2f} - {end:2.2f}'.format(start=starts[i], end=ends[i]))
    return labels


# Merge separate motif groups by iterating through the timestamps.
def merge_motifs(start, end, labels):
    # Merge motifs present in labels
    num_motifs = labels.shape[0]
    # merge_seg = np.empty([0, 3])
    merge_start = []
    merge_end = []
    merge_labels = []

    cur_start = start[0]
    cur_end = end[0]
    cur_label = labels[0]
    for i in range(num_motifs):
        if cur_end > start[i]:
            # Case 1: Adjacent segments have the same cluster group
            # Shift merge end forward
            if cur_label == labels[i]:
                cur_end = end[i]
            # Case 2: Adjacent segments have different cluster groups
            # End the current motif merge and start a new one
            else:
                # merge_seg = np.vstack((merge_seg, [cur_start, start[i], cur_label]))
                merge_start.append(cur_start)
                merge_end.append(start[i])
                merge_labels.append(cur_label)
                cur_start = start[i]
                cur_end = end[i]
                cur_label = labels[i]
        # Case 3: Adjacent segments are disjoint
        else:
            # merge_seg = np.vstack((merge_seg, [cur_start, cur_end, cur_label]))
            merge_start.append(cur_start)
            merge_end.append(cur_end)
            merge_labels.append(cur_label)
            cur_start = start[i]
            cur_end = end[i]
            cur_label = labels[i]

    merge_start.append(cur_start)
    merge_end.append(cur_end)
    merge_labels.append(cur_label)
    # Find any unexplained sections of audio within motifs
    # for i in range(len(merge_labels)-1):
    #     if merge_end[i] < merge_start[i + 1]:
    #         merge_start.insert(i, merge_end[i])
    #         merge_end.insert(i, merge_start[i])
    #         merge_labels.insert(i, -1)
    merge_seg = np.array((merge_start, merge_end, merge_labels))
    print(merge_seg.T)
    return merge_seg[0, :], merge_seg[1, :], merge_seg[2, :]


def main():
    # Create synthetic audio sequence
    return


if __name__ == '__main__':
    main()
