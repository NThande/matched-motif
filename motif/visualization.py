import matplotlib.pyplot as plt
import librosa.display
import numpy as np


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


def plot_similarity_matrix(similarity_matrix, tick_step=3):
    num_windows = similarity_matrix.shape[0]
    plt.figure()
    # plt.title("Self-Similarity Matrix Using Matched Filter")
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


def plot_window_length_comparision(windows_coll, matches_coll, sliding_lengths, labels=None):
    plt.figure()
    ax = plt.gca()
    plt.title("Similarity Scores for Various Window Lengths")
    plt.xlabel("Snippet Starting Point (s)")
    plt.ylabel("Similarity Score")
    for i in range(0, len(windows_coll)):
        plt.plot(windows_coll[i], matches_coll[i], 'C{}-'.format(i), label="{} s".format(str(sliding_lengths[i])))

    if labels is not None:
        ax = plot_add_motif_labels(ax, labels)

    plt.grid()
    plt.legend()
    return ax


# Plot Spectrogram peaks
def plot_peaks(peaks):
    if peaks is None:
        return
    plt.figure()
    plt.plot(peaks[:, PEAK_TIME_IDX], peaks[:, FREQ_IDX], 'rx')
    plt.title('STFT Peaks')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)


# Plot spectogram peaks and one pair from every inc pairs
def plot_pairs(peaks, pairs, inc=50):
    if peaks is None or pairs is None:
        return

    pair_mask = np.zeros(pairs.shape[0]).astype(int)
    for i in range(0, pairs.shape[0]):
        if i % inc == 0: pair_mask[i] = i
    pruned = pairs[pair_mask, :]

    plt.figure()
    plt.plot(peaks[:, PEAK_TIME_IDX], peaks[:, FREQ_IDX], 'rx')
    plt.plot([pruned[:, PAIR_TIME_IDX], pruned[:, PAIR_TIME_IDX + 1]],
             [pruned[:, FREQ_IDX], pruned[:, FREQ_IDX + 1]], 'b-')

    plt.plot(pruned[:, PAIR_TIME_IDX], pruned[:, FREQ_IDX], 'kx')
    plt.plot(pruned[:, PAIR_TIME_IDX + 1], pruned[:, FREQ_IDX + 1], 'k*')
    plt.title('Peak Pairs')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)


# Plot everything on the same plot
def plot_spect_peaks_pairs(freqs, times, spectrogram, peaks, pairs, inc=50):
    plt.figure()
    plt.pcolormesh(times, freqs, np.abs(spectrogram), vmin=0, vmax=5)
    plt.title('Spectogram Magnitude with Peaks & Sample Pairs')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    if peaks is None:
        return
    plt.plot(peaks[:, PEAK_TIME_IDX], peaks[:, FREQ_IDX], 'rx', label='Peaks')

    if pairs is None:
        return
    pair_mask = np.zeros(pairs.shape[0]).astype(int)
    for i in range(0, pairs.shape[0]):
        if i % inc == 0: pair_mask[i] = i
    pruned = pairs[pair_mask, :]

    plt.plot([pruned[:, PAIR_TIME_IDX], pruned[:, PAIR_TIME_IDX + 1]],
             [pruned[:, FREQ_IDX], pruned[:, FREQ_IDX + 1]], 'w-')
    plt.show(block=False)


def show():
    plt.show()