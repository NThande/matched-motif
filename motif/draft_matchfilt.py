from scipy.io.wavfile import write
import fingerprint as finPrint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Apply a sliding window of a part of the song to the rest of the song
def matched_filter(song, fs, window_length=2, write_name=None, to_plot=True, get_info=False, labels=None):
    sound = song.reshape(-1, 1)
    r = fs
    num_samples = sound.shape[0]
    song_length = np.ceil(num_samples / r)
    snap_num = int(song_length - window_length)
    snap_windows = np.arange(0, snap_num)
    snap_matches = np.zeros(snap_num)
    snap_length = window_length * r

    for i in range(0, snap_num):
        snap_start = int(i * r)
        snap_end = int((i + window_length) * r)
        if snap_end > num_samples:
            snap_end = num_samples

        snap = sound[snap_start: snap_end]
        snap_vect = np.zeros(snap_num)
        k = 0
        j = 0
        while k + snap_length < num_samples:
            sound_snap = sound[k: k + snap_length]
            snap_vect[j] = np.dot(snap.T, sound_snap)
            snap_matches[i] += np.dot(snap.T, sound_snap)
            k += r
            j += 1
        if to_plot and get_info and (i % int(snap_num / 5) == 0):
            plt.rcParams["figure.figsize"] = [16, 4]
            fig = plt.figure()
            fig.set_size_inches(13.5, 10.5, forward=True)
            plt.title("Matched Filter Results for Window {}s - {}s".format(i, i + window_length))
            plt.plot(np.arange(0, snap_vect.shape[0]), snap_vect, 'gx-')
            plt.grid()
            plt.xlabel("Seconds (s)")
            plt.ylabel("Dot Product Value")

        print("Completed Window {} / {}".format(i, snap_num))
    snap_matches = snap_matches / np.max(snap_matches)
    max_samp_idx = np.argmax(snap_matches)
    max_samp_num = snap_windows[max_samp_idx]
    max_sample = sound[max_samp_num * r + 1: (max_samp_num * r) + int((window_length) * r)]

    if to_plot:
        plt.rcParams["figure.figsize"] = [16, 4]
        plt.figure()
        plt.plot(snap_windows, snap_matches, 'rx-')
        plt.xlabel("Snippet Starting Point (s)")
        plt.ylabel("Snippet Similarity")
        plt.title("Self-Similarity Using a Matched Filter".format(window_length))

        if labels is not None:
            axes = plt.gca()
            axes.set_xlim(axes.get_xlim()[0] - 1, axes.get_xlim()[1] + 1)
            for i in range(0, labels.shape[0] - 1):
                plt.axvspan(labels.Time[i], labels.Time[i + 1], alpha=0.2, color=labels.Color[i],
                            linestyle='-.', label='Motif {}'.format(labels.Event[i]))
            plt.grid()
            plt.legend(framealpha=1.0)

    if write_name is not None:
        write(write_name, r, max_sample)

    if get_info:
        return max_sample, snap_windows, snap_matches
    print("Matched Filter complete")
    return max_sample


path = './main/bin/unique/'
file_type = '.wav'
audio_file = 't3_train'
label_file = '_labels.csv'
audio, r = finPrint.read_audio(path + audio_file + file_type)
audio_labels = pd.read_csv(path + audio_file + label_file)
plt.rc('font', size=15)  # controls default text sizes
matched_filter(audio, r, to_plot=True, window_length=2, labels=audio_labels, get_info=True)
fig = plt.gcf()
fig.set_size_inches(13.5, 10.5, forward=True)
plt.show()
