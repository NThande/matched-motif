import numpy as np
import fileutils
import shazam
import segmentation as seg
import visutils as vis


def thumbnail(audio, fs, length, include_self=False, seg_method='regular'):
    # Segment the audio
    segments_in_seconds = seg.segment(audio, fs, length=length, method=seg_method)
    segments = segments_in_seconds * fs

    # Calculate the self-similarity matrix
    num_segments = segments_in_seconds.shape[0]
    similarity = np.zeros((num_segments, num_segments))
    segment_fp = []

    # Pre-compute Shazam fingerprints
    for i in range(0, num_segments):
        cur_start = int(segments[i])
        cur_end = int(segments[i] + (length * fs))
        cur_sound = audio[cur_start: cur_end]
        cur_fp, _, _ = shazam.fingerprint(cur_sound)
        segment_fp.append(cur_fp)

    for i in range(0, num_segments):
        # Calculate similarity (forwards only, for a symmetric measure)
        cur_matches = np.zeros(num_segments)
        for j in range(i, num_segments):

            if include_self is False and i == j:
                continue

            cur_matches[j] = shazam.hash_search(segment_fp[i], segment_fp[j])

        similarity[:, i] = cur_matches

    # Row normalization (after similarity calculation)
    similarity = 0.5 * (similarity.T + similarity)
    similarity = seg.row_normalize(segments, similarity)

    # Identify the thumbnail
    sim_curve = np.sum(similarity, axis=1) / np.sum(similarity)
    thumb_idx = np.argmax(sim_curve)
    thumb_start = int(segments[thumb_idx])
    thumb_end = int(segments[thumb_idx] + (length * fs))
    thumb = audio[thumb_start: thumb_end]

    return thumb, sim_curve, segments_in_seconds, similarity


# Apply a sliding window of a part of the song to the rest of the song
def thumbnail_linear(audio, fs, audio_pairs, window_length=2, overlap=0.5):
    segments_in_seconds = seg.segment_regular(audio, fs, length=window_length, overlap=overlap)
    segments = segments_in_seconds * fs
    num_segments = segments_in_seconds.shape[0]
    seg_matches = np.zeros(num_segments)

    # Identify similarity of segment to the entire piece of audio
    for i in range(0, num_segments - 1):
        snap_start = int(segments[i])
        snap_end = int(segments[i + 1])
        snap = audio[snap_start: snap_end]
        _, seg_pairs, _ = shazam.fingerprint(snap)
        segment_hits = shazam.linear_search(audio_pairs, seg_pairs)
        seg_matches[i] = np.average(segment_hits)

    # Normalize and identify maximum window
    seg_matches = seg_matches / np.max(seg_matches)
    thumb_idx = np.argmax(seg_matches)
    thumb_segment = audio[int(segments_in_seconds[thumb_idx] * fs): int(segments_in_seconds[thumb_idx] * fs)]

    return thumb_segment, seg_matches, segments_in_seconds


# Run the thumbnailing strategy for different window lengths
def thumbnail_multi(audio, fs, length_vect):
    num_experiments = length_vect.shape[0]
    matches_list = []
    segments_list =[]
    for i in range(0, num_experiments):
        _, matches, segments, _ = thumbnail(audio, fs, length=length_vect[i])
        matches_list.append(matches)
        segments_list.append(segments)
    return matches_list, segments_list


def main():
    name = 't3_train'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    audio_labels = fileutils.load_labels(name, label_dir=directory)

    # audio_hashes, audio_pairs, _ = shazam.fingerprint(audio)
    # _, matches, segments = thumbnail_shazam(audio, fs, audio_pairs)
    # ax = vis.plot_similarity_curve(matches, segments, labels=audio_labels)
    # ax.set_title('Similarity Curve for {}'.format(name))

    thumb, similarity, segments, sim_matrix = thumbnail(audio, fs,
                                                        seg_method='onset',
                                                        length=2)

    ax = vis.plot_similarity_matrix(sim_matrix)
    ax.set_title('Regular Segmentation Similarity Matrix')
    ax = vis.plot_similarity_curve(similarity, segment_times=segments, labels=audio_labels)
    ax.set_title('Regular Segmentation Similarity')
    ax.legend()
    # ax = vis.plot_window_overlap(segments, np.ones(segments.shape) * 2, tick_step=3)
    # ax.set_title('Regular Segmentation Overlap')
    # ax.grid()
    vis.show()

    return


if __name__ == '__main__':
    main()
