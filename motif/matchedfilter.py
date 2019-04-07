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
def thumbnail(audio, fs, length, include_self=False, seg_method='regular', **kwargs):
    # Segment the audio
    segments = seg.segment(audio, fs, length=length, method=seg_method)
    segments_in_samples = segments * fs

    # Calculate the self-similarity matrix
    num_windows = segments.shape[0]
    similarity = np.zeros((num_windows, num_windows))

    for i in range(0, num_windows):
        cur_start = int(segments_in_samples[i])
        cur_end = int(segments_in_samples[i] + (length * fs))
        cur_sound = audio[cur_start: cur_end]

        # Calculate similarity with matched filter
        cur_matches = np.abs(matched_filter(cur_sound, audio, segments_in_samples))
        if include_self is False:
            cur_matches[i] = 0
        # Row normalization
        similarity[:, i] = cur_matches / np.sum(cur_matches)

    # Identify the thumbnail
    sim_curve = np.sum(similarity, axis=1) / np.sum(similarity)
    # print(np.sum(similarity, axis=0))
    thumb_idx = np.argmax(sim_curve)
    thumb_start = int(segments_in_samples[thumb_idx])
    thumb_end = int(segments_in_samples[thumb_idx] + (length * fs))
    thumb = audio[thumb_start: thumb_end]

    # Flatten out low similarity values
    return thumb, sim_curve, segments, similarity


def main():
    # Run example with regular segmentation
    name = 't3_train'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    audio_labels = fileutils.load_labels(name, label_dir=directory)

    thumb, similarity, segments, sim_matrix = thumbnail(audio, fs, seg_method='regular', length=2)

    ax = vis.plot_similarity_matrix(sim_matrix)
    ax.set_title('Regular Segmentation Similarity Matrix')
    ax = vis.plot_similarity_curve(similarity, segment_times=segments, labels=audio_labels)
    ax.set_title('Regular Segmentation Similarity')
    ax.legend()
    ax = vis.plot_window_overlap(segments, np.ones(segments.shape) * 2, tick_step=3)
    ax.set_title('Regular Segmentation Overlap')
    ax.grid()

    vis.show()


if __name__ == '__main__':
    main()
