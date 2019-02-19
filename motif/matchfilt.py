import librosa as lb
import numpy as np
import fileutils
import visualization as vis


# Applies a sliding window matched filter using ref as the reference signal and windows of
# input_sig as the test signal at intervals of step.
def windowed_matched_filter(ref, input_sig, step):
    num_windows = int((input_sig.shape[0] - ref.shape[0]) / step)
    length = ref.shape[0]
    match_results = np.zeros(num_windows)
    for j in range(0, num_windows):
        cur_sig = input_sig[j * step: (j * step) + length]
        match_results[j] = np.dot(ref.T, cur_sig)
    return match_results


# Using a series of windowed matched filters of length window_length (in seconds)
# on sound with sampling frequency fs, identify the audio thumbnail
def mf_thumbnail(sound, fs, window_length, window_step):
    sound_length = np.ceil(sound.shape[0] / fs)
    num_windows = int((sound_length - window_length) / window_step)
    window_similarity = np.zeros(num_windows)
    window_matches = np.zeros((num_windows - 1, num_windows))
    step_samples = window_step * fs

    # Calculate the matched filters
    for i in range(0, num_windows):
        cur_start = i * step_samples
        cur_end = cur_start + (window_length * fs)
        cur_sound = sound[cur_start: cur_end]
        cur_matches = np.abs(windowed_matched_filter(cur_sound, sound, step_samples))
        window_matches[:, i] = cur_matches
        window_similarity[i] = np.sum(cur_matches)
        print("Window {} / {} Complete".format(i + 1, num_windows))

    # Identify the thumbnail
    window_similarity = window_similarity / np.max(window_similarity)
    thumb_idx = np.argmax(window_similarity)
    thumb_start = thumb_idx * step_samples
    thumb_end = (thumb_idx + window_length) * step_samples
    thumbnail_sound = sound[thumb_start: thumb_end]

    return thumbnail_sound, window_similarity, window_matches


def mf_thumbnail_onset(sound, fs):
    # Detect onsets and merge segments forwards to meet minimum window length
    onsets = lb.onset.onset_detect(sound, fs, hop_length=512, units='samples', backtrack=True)
    onsets = np.append(np.insert(onsets, 0, 0), sound.shape[0])
    num_windows = onsets.shape[0] - 1
    window_similarity = np.zeros(num_windows)
    window_matches = np.zeros((num_windows - 1, num_windows))

    # Calculate the matched filters
    for i in range(0, num_windows):
        cur_start = onsets[i]
        cur_end = onsets[i + 1]
        cur_sound = sound[cur_start: cur_end]
        step_samples = int((cur_end - cur_start) / 2)
        cur_matches = np.abs(windowed_matched_filter(cur_sound, sound, step_samples))
        # window_matches[:, i] = cur_matches
        window_similarity[i] = np.sum(cur_matches)
        print("Window {} / {} Complete".format(i + 1, num_windows))

    # Identify the thumbnail
    window_similarity = window_similarity / np.max(window_similarity)
    thumb_idx = np.argmax(window_similarity)
    thumb_start = onsets[thumb_idx]
    thumb_end = onsets[thumb_idx + 1]
    thumbnail_sound = sound[thumb_start: thumb_end]

    return thumbnail_sound, window_similarity, window_matches


def main():
    name = 't3_train'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    audio_labels = fileutils.load_labels(name, label_dir=directory)

    thumbnail, similarity, sim_matrix = mf_thumbnail(audio, fs, window_length=2, window_step=1)

    vis.plot_similarity_matrix(sim_matrix)
    vis.plot_mf_similarity(similarity, window_step=1, labels=audio_labels)
    vis.plot_mf_window_layout(audio, fs, window_length=2, window_step=1)
    vis.show()


if __name__ == '__main__':
    main()
