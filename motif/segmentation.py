import fileutils
import numpy as np
import librosa as lb
import config as cfg
import pandas as pd


# Choose a segmentation method from a given input
def segment(audio, fs, length=2, overlap=cfg.OVERLAP_RATIO, method='regular'):
    segments = None
    if method is 'regular':
        segments = segment_regular(audio, fs, length=length, overlap=overlap)
    elif method is 'onset' or method is 'beat':
        segments = segment_onset(audio, fs, length=length, overlap=overlap, method=method)
    else:
        print("Unrecognized segmentation method: {}".format(method))
    return segments


# Create regular segments of fixed length with an overlap ratio in seconds
def segment_regular(audio, fs, length, overlap):
    total_length = audio.shape[0] / fs
    segment_list = []
    k = 0
    while k < (total_length - length):
        segment_list.append(k)
        k += length * (1 - overlap)
    segments = np.asarray(segment_list)
    return segments


# Detect onsets and merge segments forwards to meet minimum window length
def segment_onset(audio, fs,
                  length,
                  overlap,
                  method='onset',
                  prune=False,
                  fill_space=False):
    if method is 'onset':
        onsets = lb.onset.onset_detect(audio, fs, hop_length=cfg.WINDOW_SIZE, units='time', backtrack=True)
    elif method is 'beat':
        _, onsets = lb.beat.beat_track(audio, fs, hop_length=cfg.WINDOW_SIZE, units='time')
    else:
        return None

    if onsets[0] != 0.:
        onsets = np.insert(onsets, 0, 0)

    num_onsets = onsets.shape[0]

    # Prune onsets that are too close to the previous onset or the end of the audio
    mask = np.zeros(onsets.shape)
    if prune is False:
        mask = np.ones(onsets.shape)
    min_diff = length * (1 - overlap)
    audio_end = audio.shape[0] / fs
    prev_onset = onsets[0]
    for i in range(1, num_onsets):
        if onsets[i] + length > audio_end:
            mask[i:num_onsets] = 0
            break
        if onsets[i] - min_diff > prev_onset:
            mask[i] = 1
            prev_onset = onsets[i]

    onsets = onsets[mask == 1]

    # Add regularly spaced windows in periods without onsets
    if fill_space is True:
        num_onsets = onsets.shape[0]
        add_idx = []
        add_seg = []
        for i in range(1, num_onsets):
            if onsets[i] - length > onsets[i - 1]:
                add_onset = onsets[i - 1] + min_diff
                while add_onset < onsets[i] - min_diff:
                    add_idx.append(i)
                    add_seg.append(add_onset)
                    add_onset = add_onset + min_diff

        onsets = np.insert(onsets, add_idx, add_seg)

    return onsets


# Create labels for segments
def seg_to_label(starts, ends):
    labels = []
    for i in range(starts.shape[0]):
        labels.append('{start:2.2f} - {end:2.2f}'.format(start=starts[i], end=ends[i]))
    return labels


# Merge separate motif groups by iterating through the timestamps.
def merge_motifs(start, end, labels):
    # Merge motifs present in labels
    num_motifs = labels.shape[0]
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
                merge_start.append(cur_start)
                merge_end.append(start[i])
                merge_labels.append(cur_label)
                cur_start = start[i]
                cur_end = end[i]
                cur_label = labels[i]
        # Case 3: Adjacent segments are disjoint
        else:
            merge_start.append(cur_start)
            merge_end.append(cur_end)
            merge_labels.append(cur_label)
            cur_start = start[i]
            cur_end = end[i]
            cur_label = labels[i]

    merge_start.append(cur_start)
    merge_end.append(cur_end)
    merge_labels.append(cur_label)

    # Convert to np array
    merge_seg = np.array((merge_start, merge_end, merge_labels))
    return merge_seg[0, :], merge_seg[1, :], merge_seg[2, :]


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
    time_labels = seg_to_label(starts, ends)
    motifs = dict.fromkeys(np.unique(labels))
    for i in range(len(time_labels)):
        if motifs[labels[i]] is None:
            motifs[labels[i]] = [time_labels[i]]
        else:
            motifs[labels[i]].append(time_labels[i])
    return motifs


# Collect motifs into pandas dataframe
def motif_to_df(starts, ends, labels):
    motif_dict = {'Start':starts, 'End':ends, 'Motif':labels}
    return pd.DataFrame(motif_dict, columns=motif_dict.keys())


def motif_join(starts, ends, motif_labels):
    return


def main():
    name = 't1'
    directory = "./bin/"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    length = 2
    overlap = 0.5
    print(segment_onset(audio, fs, length=length, overlap=overlap))
    print(segment_onset(audio, fs, length=length, overlap=overlap, method='beat'))


if __name__ == '__main__':
    main()
