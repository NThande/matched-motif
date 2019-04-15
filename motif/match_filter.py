import librosa.display
import numpy as np

import fileutils
import segmentation as seg
import visutils as vis


# Applies a sliding window matched filter using ref as the reference signal and segments of
# sig as the test signal. Segment starting points are determined by the vector segments.
def matched_filter(ref, sig, segments):
    num_windows = segments.shape[0]
    length = ref.shape[0]
    match_results = np.zeros(num_windows)
    for j in range(0, num_windows):
        cur_sig = sig[int(segments[j]): int(segments[j] + length)]
        cur_sig = cur_sig[0:length]
        match_results[j] = np.dot(ref.T, cur_sig)
    return match_results


# Using a series of windowed matched filters of length window_length (in seconds)
# on sound with sampling frequency fs, identify the audio thumbnail
def thumbnail(audio, fs, length, include_self=True, seg_method='regular'):
    # Segment the audio
    segments_in_seconds = seg.segment(audio, fs, length=length, method=seg_method)
    segments = segments_in_seconds * fs

    # Calculate the self-similarity matrix
    num_segments = segments_in_seconds.shape[0]
    similarity = np.zeros((num_segments, num_segments))

    for i in range(0, num_segments):
        cur_start = int(segments[i])
        cur_end = int(segments[i] + (length * fs))
        cur_sound = audio[cur_start: cur_end]

        # Calculate similarity with matched filter
        cur_matches = np.abs(matched_filter(cur_sound, audio, segments))

        if include_self is False:
            cur_matches[i] = 0

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


def main():
    # Run example with regular segmentation
    name = 't1'
    directory = "./bin/test"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    audio_labels = fileutils.load_labels(name, label_dir=directory)

    thumb, similarity, segments, sim_matrix = thumbnail(audio, fs,
                                                        seg_method='regular',
                                                        length=2)

    ax = vis.plot_similarity_matrix(sim_matrix)
    ax.set_title('Regular Segmentation Similarity Matrix')
    ax = vis.plot_similarity_curve(similarity, segment_times=segments, labels=audio_labels)
    ax.set_title('Regular Segmentation Similarity')
    ax.legend()
    vis.show()


if __name__ == '__main__':
    main()
