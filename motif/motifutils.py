import numpy as np
import pandas as pd
from suffix_trees import STree
from collections import defaultdict

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

    return np.array(merge_start), np.array(merge_end), np.array(merge_labels, dtype=int)


# Prune motifs shorter than the min_length.
def prune_motifs(starts, ends, motif_labels, min_length=1):
    num_motifs = motif_labels.shape[0]
    prune_list = []
    for i in range(num_motifs):
        diff = ends[i] - starts[i]
        if diff < min_length:
            prune_list.append(i)
    starts = np.delete(starts, prune_list)
    ends = np.delete(ends, prune_list)
    motif_labels = np.delete(motif_labels, prune_list)
    return starts, ends, motif_labels


# Join motifs using the string-join method
def motif_join(starts, ends, motif_labels):
    # Transform labels to string
    sep = ','
    label_str = np.array_str(motif_labels).replace('[', sep).replace(']', sep).replace(' ', sep)
    label_dict = {}
    cur_relabel = np.max(motif_labels) + 1
    lrs_len = motif_labels.shape[0]

    # How do we avoid overlap?
    while lrs_len > 1:
        lrs = longest_repeated_substring(label_str)
        if lrs[0] is sep:
            lrs = lrs[1:]
        lrs_arr = np.fromstring(lrs, count=-1, sep=sep, dtype=int)
        lrs_len = lrs_arr.shape[0]

        # Is this string too short?
        if lrs_len <= 1:
            break

        label_dict[cur_relabel] = np.fromstring(lrs, count=-1, sep=sep, dtype=int)
        label_str = label_str.replace(lrs, str(cur_relabel) + sep)
        cur_relabel += 1

    label_str = label_str.replace(sep + sep, sep)
    if label_str[0] is sep:
        label_str = label_str[1:]

    # Convert segmentation to new format
    relabels = np.fromstring(label_str, count=-1, sep=sep, dtype=int)
    restarts = np.zeros(relabels.shape)
    re_ends = np.zeros(relabels.shape)
    label_idx = 0
    for i in range(relabels.shape[0]):
        seq_length = 0
        if relabels[i] in label_dict.keys():
            seq_length = label_dict[relabels[i]].shape[0] - 1

        if label_idx >= starts.shape[0]:
            restarts[i] = starts[starts.shape[0] - 1]
        else:
            restarts[i] = starts[label_idx]

        if label_idx + seq_length >= ends.shape[0]:
            re_ends[i] = ends[ends.shape[0] - 1]
        else:
            re_ends[i] = ends[label_idx + seq_length]

        label_idx += seq_length + 1
    relabels = sequence_labels(relabels)
    return restarts, re_ends, relabels


# Dynamic programming method borrowed from https://www.geeksforgeeks.org/longest-repeating-and-non-overlapping
# -substring/ Returns the longest repeating non-overlapping substring in str
def longest_repeated_substring(str):
    n = len(str)
    LCSRe = [[0 for x in range(n + 1)]
             for y in range(n + 1)]

    res = ""  # To store result
    res_length = 0  # To store length of result

    # building table in bottom-up manner
    index = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):

            # (j-i) > LCSRe[i-1][j-1] to remove
            # overlapping
            if (str[i - 1] == str[j - 1] and
                    LCSRe[i - 1][j - 1] < (j - i)):
                LCSRe[i][j] = LCSRe[i - 1][j - 1] + 1

                # updating maximum length of the
                # substring and updating the finishing
                # index of the suffix
                if (LCSRe[i][j] > res_length):
                    res_length = LCSRe[i][j]
                    index = max(i, index)

            else:
                LCSRe[i][j] = 0

    # If we have non-empty result, then insert
    # all characters from first character to
    # last character of string
    if (res_length > 0):
        for i in range(index - res_length + 1,
                       index + 1):
            res = res + str[i - 1]

    return res


# Renumber integer labels so that they appear in order in the labels list.
def sequence_labels(labels):
    unique_labels = np.unique(labels)
    label_map = dict.fromkeys(unique_labels)
    label_pos = np.zeros(np.max(unique_labels) + 1)

    for i in range(label_pos.shape[0]):
        if i in unique_labels:
            label_pos[i] = np.nonzero(labels == i)[0][0]
        else:
            label_pos[i] = labels.shape[0]

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
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3])
    # print(labels)
    # motif_join(None, None, labels)
    # labels = np.random.random_integers(0, 3, 10)
    starts = np.arange(0, 12)
    ends = starts + 1
    print(labels)
    restarts, reends, relabels = motif_join(starts, ends, labels)
    print(restarts)
    print(reends)
    simple_str = 'I am I am '
    complex_str = 'I am I am I'
    return


if __name__ == '__main__':
    main()
