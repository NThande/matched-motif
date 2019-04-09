import numpy as np
import editdistance


# For simplicity, this algorithm assumes that segments is a 2 x N array of N segment starts and ends.
# Borrowed from Chai and Vercoe.
# Returns a metric reflecting the distance between the nearest neighbors in observed_segments and reference_segments.
def boundary_distance(obs_labels, ref_labels, obs_segments, ref_segments):
    # Recall
    r = recall(obs_labels, ref_labels, obs_segments, ref_segments)
    # Number of repetitive segments in ideal structure
    s = np.unique(ref_labels).shape[0]
    return (1 - r)/s


# Returns the Levenshtein distance between the observed_labels and reference_labels.
def edit_distance(observed_labels, reference_labels):
    observed_string = np.array2string(observed_labels)
    reference_string = np.array2string(reference_labels)
    return editdistance.eval(observed_string, reference_string)


# Returns difference in number of unique labels.
def motif_count(obs_labels, ref_labels):
    obs_count = np.unique(obs_labels).shape[0]
    ref_count = np.unique(ref_labels).shape[0]
    return obs_count - ref_count


# Combines Recall and Precision methods as per the method by Masataka Goto.
def f_measure(obs_labels, ref_labels, obs_segments, ref_segments):
    # Recall
    r = recall(obs_labels, ref_labels, obs_segments, ref_segments)
    # Precision
    p = precision(obs_labels, ref_labels, obs_segments, ref_segments)
    # F-Measure
    return (2*r*p)/(r + p)


def recall(obs_labels, ref_labels, obs_segments, ref_segments):
    # Sum of length of correctly observed motifs
    shared_seg = shared_motifs(obs_labels, ref_labels, obs_segments, ref_segments)
    shared_len = sum_segments(shared_seg)
    # Sum of length of ideal motifs
    ref_len = sum_segments(ref_segments)
    return shared_len/ref_len


def precision(obs_labels, ref_labels, obs_segments, ref_segments):
    # Sum of length of correctly observed motifs
    shared_seg = shared_motifs(obs_labels, ref_labels, obs_segments, ref_segments)
    shared_len = sum_segments(shared_seg)
    # Sum of observed motifs
    obs_len = sum_segments(obs_segments)
    return shared_len/obs_len


# Produces a segmentation with only segments with the same label between the obs and ref sets.
def shared_motifs(obs_labels, ref_labels, obs_segments, ref_segments):
    share_start = []
    share_end = []
    for i in range(ref_segments.shape[1]):
        for j in range(obs_segments.shape[1]):
            # Find where the segments start to overlap
            if obs_segments[1, j] < ref_segments[0, i]:
                continue
            elif obs_segments[0, j] > ref_segments[1, i]:
                continue
            # Does this segment have the right label?
            elif obs_labels[j] != ref_labels[i]:
                continue
            # Capture the overlapping region
            start = np.maximum(obs_segments[0, j], ref_segments[0, i])
            end = np.minimum(obs_segments[1, j], ref_segments[1, i])
            share_start.append(start)
            share_end.append(end)

    share_segments = np.array((share_start, share_end))
    return share_segments


# Sum of length of segments.
def sum_segments(segments):
    time_diff = segments[1, :] - segments[0, :]
    return np.sum(time_diff)


def main():
    # Create some pseudo-segments
    test_ref_labels = np.array([0, 1, 0, 1])
    test_obs_labels = np.array([0, 1, 1, 0, 1])
    test_ref_segments = np.array(([0., 1., 2., 3.],
                                 [1., 2., 3., 4.]))
    test_obs_segments = np.array(([0., 1., 2.5, 2.75, 3.3],
                                 [1., 2.5, 2.75, 3., 4.]))
    print(edit_distance(test_ref_labels, test_obs_labels))
    print("Ref:\n", test_ref_segments)
    print(test_ref_labels)
    print("Obs:\n", test_obs_segments)
    print(test_obs_labels)
    shared_segments = shared_motifs(test_obs_labels, test_ref_labels, test_obs_segments, test_ref_segments)
    print("Shared:\n", shared_segments)
    print("Shared Length: \n", sum_segments(shared_segments))
    return


if __name__ == '__main__':
    main()
