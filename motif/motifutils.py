import numpy as np
import pandas as pd
import segmentation as seg


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
    motif_dict = {'Start':starts, 'End':ends, 'Motif':labels}
    return pd.DataFrame(motif_dict, columns=motif_dict.keys())


def df_to_motif(labels_df):
    labels = labels_df['Event'].values
    segments = labels_df['Time'].values
    num_segments = segments.shape[0]

    starts = segments[:num_segments - 1]
    ends = segments[1:]
    motif_labels = labels[:num_segments - 1].astype(int)
    return starts, ends, motif_labels


def motif_join(starts, ends, motif_labels):
    return


def main():
    return


if __name__ == '__main__':
    main()
