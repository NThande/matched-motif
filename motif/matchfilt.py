import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


# Plot the similarity used to calculate a thumbnail from mf_thumbnail.
def plot_mf_similarity(window_similarity, window_step, labels=None):
    window_start = np.zeros(window_similarity.shape)
    for i in range(0, window_start.shape[0]):
        window_start[i] = i * window_step

    plt.figure()
    ax = plt.gca()
    ax.plot(window_start, window_similarity, 'rx-')
    plt.xlabel("Window Starting Point (s)")
    plt.ylabel("Window Similarity")
    plt.title("Self-Similarity Using a Matched Filter")
    if labels is not None:
        ax = plot_add_motif_labels(ax, labels)
    return ax


# Add hand-labeled motifs to a similarity plot for a mf thumbnail
def plot_add_motif_labels(ax, labels):
    ax.set_xlim(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1)
    for i in range(0, labels.shape[0] - 1):
        plt.axvspan(labels.Time[i], labels.Time[i + 1], alpha=0.2, color=labels.Color[i],
                    linestyle='-.', label='Motif {}'.format(labels.Event[i]))
    ax.grid()
    return ax


def plot_mf_similarity_matrix(similarity_matrix, tick_step=3):
    num_windows = similarity_matrix.shape[0]
    plt.figure()
    plt.title("Self-Similarity Matrix Using Matched Filter")
    plt.xlabel("Window #")
    plt.ylabel("Window #")
    ax = plt.gca()
    ax.xaxis.set_ticks(np.arange(0, num_windows, tick_step))
    ax.yaxis.set_ticks(np.arange(0, num_windows, tick_step))
    plt.imshow(similarity_matrix)
    plt.colorbar()
    return ax


def plot_mf_window_layout(sound, fs, window_length, window_step, tick_step=1):
    sound = sound.reshape(-1, 1)
    sound_length = np.ceil(sound.shape[0] / fs)
    num_windows = int((sound_length - window_length) / window_step)

    plt.figure()
    plt.title("Window layout for {}s windows at {}s intervals".format(window_length, window_step))
    plt.xlabel("Seconds")
    plt.ylabel("Window #")
    ax = plt.gca()
    ax.set_ylim(-1, num_windows)
    ax.set_xlim(0, num_windows)
    ax.xaxis.set_ticks(np.arange(0, num_windows, tick_step))
    ax.yaxis.set_ticks(np.arange(0, num_windows - 1, tick_step))
    ax.grid()
    for i in range(0, num_windows - 1):
        line_start = i / num_windows
        line_end = (i + window_length) / num_windows
        ax.axhline(i, line_start, line_end)
        ax.plot([i, i + window_length], [i, i], 'rx')
    return ax


path = './main/bin/unique/'
file_type = '.wav'
audio_file = 't3_train'
label_file = '_labels.csv'
audio, fs = lb.load(path + audio_file + file_type)
audio_labels = pd.read_csv(path + audio_file + label_file)
thumbnail, similarity, sim_matrix = mf_thumbnail(audio, fs, window_length=2, window_step=1)
# thumbnail, similarity, sim_matrix = mf_thumbnail_onset(audio, fs, window_length=2)
plot_mf_similarity_matrix(sim_matrix)
plot_mf_similarity(similarity, window_step=1, labels=audio_labels)
plot_mf_window_layout(audio, fs, window_length=2, window_step=1)
plt.show()
