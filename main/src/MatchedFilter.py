from scipy.io.wavfile import write
import FingerPrint as finPrint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Apply a sliding window of a part of the song to the rest of the song
def matched_filter(song, fs, window_length=2, write_name=None, to_plot=True, get_info=False, labels=None):
    sound = song.reshape(-1, 1)
    # r = song_fp.fs
    r = fs
    num_samples = sound.shape[0]
    song_length = np.ceil(num_samples / r)
    snap_num = int(song_length - window_length)
    snap_windows = np.arange(0, snap_num)
    snap_matches = np.zeros(snap_num)
    # snap_hits = np.zeros(song_fp.pairs.shape[0])
    snap_length = window_length * r

    for i in range(0, snap_num):
        snap_start = int(i * r)
        snap_end = int((i + window_length) * r)
        if snap_end > num_samples:
            snap_end = num_samples

        snap = sound[snap_start: snap_end]
        k = 0
        while k + snap_length < num_samples:
            sound_snap = sound[k : k + snap_length]
            snap_matches[i] += np.dot(snap.T, sound_snap)
            k += r
        # snap_fp = finPrint.FingerPrint(snap, r)
        # snap_peaks, snap_pairs = snap_fp.generate_fingerprint()
        # match_vect = np.zeros(snap_pairs.shape[0])

        # for j in range(0, snap_pairs.shape[0]):
        #     match_idx, match_vect[j] = song_fp.search_for_pair(snap_pairs[j])
        #     snap_hits[match_idx] += 1
        # snap_matches[i] = np.average(match_vect)
        print("Completed Window {} / {}".format(i, snap_num))
    snap_matches = snap_matches / np.max(snap_matches)
    max_samp_idx = np.argmax(snap_matches)
    max_samp_num = snap_windows[max_samp_idx]
    max_sample = sound[max_samp_num * r + 1: (max_samp_num * r) + int((window_length) * r)]

    if to_plot:
        plt.figure()
        plt.plot(snap_windows, snap_matches, 'rx-')
        plt.xlabel("Snippet Starting Point (s)")
        plt.ylabel("Snippet Similarity")
        plt.title("Self-Similarity Using a Matched Filter".format(window_length))

        if labels is not None:
            axes = plt.gca()
            y_max = axes.get_ylim()[1]
            axes.set_xlim(axes.get_xlim()[0] - 1, axes.get_xlim()[1] + 1)
            for i in range(0, labels.shape[0] - 1):
                plt.axvspan(labels.Time[i], labels.Time[i+1], alpha=0.2,  color=labels.Color[i],
                            linestyle='-.', label='Motif {}'.format(labels.Event[i]))
                # plt.annotate(labels.Event[i], xy=(labels.Time[i], 0.9 * y_max),
                #              xytext=(5, 0), textcoords='offset points', rotation=45)
            plt.grid()
            # plt.legend(framealpha=1.0)

    if write_name is not None:
        write(write_name, r, max_sample)

    if get_info:
        return max_sample, snap_windows, snap_matches
    print("Matched Filter complete")
    return max_sample


path = './main/bin/unique/'
file_type = '.wav'
audio_file = 'hello_train'
label_file = '_labels.csv'
audio, r = finPrint.read_audio(path + audio_file + file_type)
audio_labels = pd.read_csv(path + audio_file + label_file)
# plt.rc('font', size=15)          # controls default text sizes
# plt.rc('axes', titlesize=22)     # fontsize of the axes title
# plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
# plt.rc('legend', fontsize=20)    # legend fontsize
# plt.rc('figure', titlesize=72)  # fontsize of the figure title
matched_filter(audio, r, to_plot=True, window_length=2, labels=audio_labels)
# plt.figure(num=1, figsize=(8,6))
# fig = plt.gcf()
# fig.set_size_inches(13.5, 10.5, forward=True)
# plt.rcParams["figure.figsize"] = [16,4]
plt.show()