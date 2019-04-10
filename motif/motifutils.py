import numpy as np
import pandas as pd
from suffix_trees import STree

import config as cfg
import segmentation as seg

START_IDX = cfg.START_IDX
END_IDX = cfg.END_IDX
LABEL_IDX = cfg.LABEL_IDX


# Merge separate motif groups by iterating through the timestamps.
def merge_motifs(starts, ends, labels):
    # Merge motifs present in labels
    num_motifs = labels.shape[0]
    merge_start = []
    merge_end = []
    merge_labels = []

    cur_start = starts[0]
    cur_end = ends[0]
    cur_label = labels[0]
    for i in range(num_motifs):
        if cur_end > starts[i]:
            # Case 1: Adjacent segments have the same cluster group
            # Shift merge end forward
            if cur_label == labels[i]:
                cur_end = ends[i]
            # Case 2: Adjacent segments have different cluster groups
            # End the current motif merge and start a new one
            else:
                merge_start.append(cur_start)
                merge_end.append(starts[i])
                merge_labels.append(cur_label)
                cur_start = starts[i]
                cur_end = ends[i]
                cur_label = labels[i]
        # Case 3: Adjacent segments are disjoint
        else:
            merge_start.append(cur_start)
            merge_end.append(cur_end)
            merge_labels.append(cur_label)
            cur_start = starts[i]
            cur_end = ends[i]
            cur_label = labels[i]

    merge_start.append(cur_start)
    merge_end.append(cur_end)
    merge_labels.append(cur_label)

    return np.array(merge_start), np.array(merge_end), np.array(merge_labels)


# Join motifs using the string-join method
def motif_join(starts, ends, motif_labels):
    # Transform labels to string
    num_motifs = motif_labels.shape[0]
    labels_str = np.array_str(motif_labels)
    labels_str = labels_str.replace('[', '').replace(']', '')
    labels_sfx = STree.STree(labels_str)
    print(labels_str)

    i = 0
    cur_seq = labels_str[i]
    prev_seq = ''
    prev_matches = []
    cur_label = np.max(motif_labels) + 1
    print(cur_label)
    while i < num_motifs - 1:
        matches = labels_sfx.find_all(cur_seq)
        i += 1
        if len(matches) > 1:
            # Check if all segments overlap
            no_overlap = False
            for j in range(len(matches) - 1):
                if matches[j + 1] - matches[j] >= len(cur_seq):
                    no_overlap = True
                    break
            if no_overlap:
                prev_seq = cur_seq
                cur_seq += labels_str[i]
            else:
                labels_str = labels_str.replace(prev_seq, str(cur_label) + ' ')
                cur_label += 1
                prev_seq = ''
                cur_seq += labels_str[i]
        else:
            if prev_seq is not '':
                labels_str = labels_str.replace(prev_seq, str(cur_label) + ' ')
                cur_label += 1
                return
            cur_seq = labels_str[i]
        print("{}: {}".format(i, matches))
        print(labels_str)

    return


# Renumber integer labels so that they appear in order in the labels list.
def sequence_labels(labels):
    unique_labels = np.unique(labels)
    label_map = dict.fromkeys(unique_labels)
    label_pos = np.zeros(unique_labels.shape[0])

    for i in unique_labels:
        label_pos[i] = np.nonzero(labels == i)[0][0]

    new_order = np.argsort(label_pos)
    for i in range(new_order.shape[0]):
        label_map[new_order[i]] = i

    for i in range(labels.shape[0]):
        labels[i] = label_map[labels[i]]

    return labels


# Collect motifs into a single output dictionary
def motif_to_dict(starts, ends, labels):
    time_labels = seg.seg_to_label(starts, ends)
    motif_dict = dict.fromkeys(np.unique(labels))
    for i in range(len(time_labels)):
        if motif_dict[labels[i]] is None:
            motif_dict[labels[i]] = [time_labels[i]]
        else:
            motif_dict[labels[i]].append(time_labels[i])
    return motif_dict


# Collect motifs into pandas dataframe
def motif_to_df(starts, ends, labels):
    motif_dict = {'Start': starts, 'End': ends, 'Motif': labels}
    return pd.DataFrame(motif_dict, columns=motif_dict.keys())


def df_to_motif(labels_df):
    labels = labels_df['Event'].values
    segments = labels_df['Time'].values
    num_segments = segments.shape[0]

    starts = segments[:num_segments - 1]
    ends = segments[1:]
    motif_labels = labels[:num_segments - 1].astype(int)
    return starts, ends, motif_labels


# Pack and unpack segments and their labels into a single array
def pack_motif(starts, ends, motif_labels):
    return np.array((starts, ends, motif_labels))


def unpack_motif(motifs):
    return motifs[START_IDX], motifs[END_IDX], motifs[LABEL_IDX]


def main():
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 3, 0, 2, 3, 4])
    print(labels)
    motif_join(None, None, labels)
    return


if __name__ == '__main__':
    main()
