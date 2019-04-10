import editdistance
import numpy as np
import config as cfg

START_IDX = cfg.START_IDX
END_IDX = cfg.END_IDX
LABEL_IDX = cfg.LABEL_IDX


# For simplicity, this algorithm assumes that segments is a 3 x N array of N segment starts, ends, and labels.
# Borrowed from Chai and Vercoe.
# Returns a metric reflecting the distance between the nearest neighbors in observed_segments and reference_segments.
def boundary_distance(obs_motifs, ref_motifs):
    # Recall
    r = recall(obs_motifs, ref_motifs)
    # Number of repetitive segments in ideal structure
    s = np.unique(ref_motifs[LABEL_IDX]).shape[0]
    return (1 - r) / s


# Returns the Levenshtein distance between the observed_labels and reference_labels.
def edit_distance(observed_labels, reference_labels):
    observed_string = np.array_str(observed_labels)
    reference_string = np.array_str(reference_labels)
    return editdistance.eval(observed_string, reference_string)


# Returns difference in number of unique labels.
def motif_count(obs_labels, ref_labels):
    obs_count = np.unique(obs_labels).shape[0]
    ref_count = np.unique(ref_labels).shape[0]
    return obs_count - ref_count


# Combines Recall and Precision methods as per the method by Masataka Goto.
def f_measure(obs_motifs, ref_motifs):
    # Recall
    r = recall(obs_motifs, ref_motifs)
    # Precision
    p = precision(obs_motifs, ref_motifs)
    # F-Measure
    return (2 * r * p) / (r + p)


def recall(obs_motifs, ref_motifs):
    # Sum of length of correctly observed motifs
    shared_seg = shared_motifs(obs_motifs, ref_motifs)
    shared_len = sum_segments(shared_seg)
    # Sum of length of ideal motifs
    ref_len = sum_segments(ref_motifs)
    return shared_len / ref_len


def precision(obs_motifs, ref_motifs):
    # Sum of length of correctly observed motifs
    # Sum of length of correctly observed motifs
    shared_seg = shared_motifs(obs_motifs, ref_motifs)
    shared_len = sum_segments(shared_seg)
    # Sum of length of observed motifs
    obs_len = sum_segments(obs_motifs)
    return shared_len / obs_len


# Produces a segmentation with only segments with the same label between the obs and ref sets.
def shared_motifs(obs_motifs, ref_motifs):
    share_start = []
    share_end = []
    for i in range(ref_motifs.shape[1]):
        for j in range(obs_motifs.shape[1]):
            # Find where the segments start to overlap
            if obs_motifs[END_IDX, j] < ref_motifs[START_IDX, i]:
                continue
            elif obs_motifs[START_IDX, j] > ref_motifs[END_IDX, i]:
                continue
            # Does this segment have the right label?
            elif obs_motifs[LABEL_IDX, j] != ref_motifs[LABEL_IDX, i]:
                continue
            # Capture the overlapping region
            start = np.maximum(obs_motifs[START_IDX, j], ref_motifs[START_IDX, i])
            end = np.minimum(obs_motifs[END_IDX, j], ref_motifs[END_IDX, i])
            if end > start:
                share_start.append(start)
                share_end.append(end)

    share_segments = np.array((share_start, share_end))
    return share_segments


# Sum of length of segments.
def sum_segments(segments):
    time_diff = segments[END_IDX, :] - segments[START_IDX, :]
    return np.sum(time_diff)


def main():
    # Create some pseudo-segments
    ref_motifs = np.array(([0., 1., 2., 3.],
                           [1., 2., 3., 4.],
                           [0, 1, 0, 1]))
    obs_motifs = np.array(([0., 1., 2.5, 2.75, 3.3],
                           [1., 2.5, 2.75, 3., 4.],
                           [0, 1, 1, 0, 1]))
    print("Edit Distance: ", edit_distance(ref_motifs[LABEL_IDX], obs_motifs[LABEL_IDX]))
    print("Ref:\n", ref_motifs)
    print("Obs:\n", obs_motifs)
    shared_segments = shared_motifs(obs_motifs, ref_motifs)
    print("Shared:\n", shared_segments)
    print("Shared Length: \n", sum_segments(shared_segments))
    return


if __name__ == '__main__':
    main()
