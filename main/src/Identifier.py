from scipy.io.wavfile import write
import FingerPrint as finPrint
import matplotlib.pyplot as plt
import numpy as np


# Test basic Shazam-style identification
def basic_search(song_fp, sample_fp):
    match_vect = np.zeros(sample_fp.pairs.shape[0])
    for i in range(0, sample_fp.pairs.shape[0]):
        _, match_vect[i] = song_fp.search_for_pair(sample_fp.pairs[i])
    matches = np.average(match_vect)
    return matches


# Search for a sample using different windows of the song centered on a point
def shrinking_search(song_fp, sample_fp, start_time, center_time, to_plot=True):
    sound = song_fp.sound
    r = song_fp.fs

    snip_start = start_time
    if (snip_start < 0):
        snip_start = 0
    snip_end = 2 * center_time - start_time
    if (snip_end > sound.shape[0] / r):
        snip_end = sound.shape[0] / r

    samp_hits = np.zeros(song_fp.pairs.shape[0])
    num_iter = int(np.floor((snip_end - snip_start) / 2))
    snip_lengths = np.zeros(num_iter)
    snip_matches = np.zeros(num_iter)

    for i in range(0, num_iter):
        snippet_fp = finPrint.FingerPrint(sound[snip_start * r: snip_end * r], Fs=r)
        snippet_fp.generate_fingerprint()
        match_vect = np.zeros(sample_fp.pairs.shape[0])
        for j in range(0, sample_fp.pairs.shape[0]):
            match_idx, match_vect[j] = snippet_fp.search_for_pair(sample_fp.pairs[j])
            samp_hits[match_idx] += 1
        if (snip_start < center_time):
            snip_start += 1
        if (snip_end > center_time):
            snip_end -= 1
        snip_matches[i] = np.average(match_vect)
        snip_lengths[i] = snip_end - snip_start

    if to_plot:
        plt.figure()
        plt.plot(snip_lengths, snip_matches, 'rx-')
        plt.xlabel("Snippet Length")
        plt.ylabel("Number of matches in database")
        plt.title("Number of matches for fixed snippet vs length of overall fingerprint track")

    return samp_hits


# Apply a sliding window of a part of the song to the rest of the song
def autowindow_search(song_fp, window_length=2, write_name=None, to_plot=True, get_info=False):
    sound = song_fp.sound
    r = song_fp.fs
    song_length = np.ceil(sound.shape[0] / r)
    snap_num = int(song_length - window_length)
    snap_windows = np.arange(0, snap_num)
    snap_matches = np.zeros(snap_num)
    snap_hits = np.zeros(song_fp.pairs.shape[0])
    for i in range(0, snap_num):
        snap_start = int(i * r)
        snap_end = int((i + window_length) * r)
        if snap_end > sound.shape[0]:
            snap_end = sound.shape[0]
        snap = sound[snap_start: snap_end]
        snap_fp = finPrint.FingerPrint(snap, r)
        snap_peaks, snap_pairs = snap_fp.generate_fingerprint()
        match_vect = np.zeros(snap_pairs.shape[0])
        for j in range(0, snap_pairs.shape[0]):
            match_idx, match_vect[j] = song_fp.search_for_pair(snap_pairs[j])
            snap_hits[match_idx] += 1
        snap_matches[i] = np.average(match_vect)

    max_samp_idx = np.argmax(snap_matches)
    max_samp_num = snap_windows[max_samp_idx]
    max_sample = sound[max_samp_num * r + 1: (max_samp_num * r) + int((window_length) * r)]

    if to_plot:
        plt.figure()
        plt.plot(snap_windows, snap_matches, 'rx-')
        plt.xlabel("Snapshot Starting Point (s)")
        plt.ylabel("Number of matches in database")
        plt.title("Average number of matches for Fixed Length = {} Sliding Window".format(window_length))

        plt.figure()
        plt.plot(song_fp.pairs[:, finPrint.PAIR_TIME_IDX], snap_hits, 'rx', label='Pair Start Time')
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Number of matches")
        plt.title("Matches per pair using Fixed Length = {} Sliding Window".format(window_length))
        plt.show(block=False)

    if write_name is not None:
        write(write_name, r, max_sample)

    if get_info:
        return max_sample, snap_windows, snap_matches, snap_hits
    return max_sample


path = './main/bin/unique/'
file_type = '.wav'
audio_file = 't3_train'
snapshot_file = 't3_test'

audio, r = finPrint.read_audio(path + audio_file + file_type)
audio_fp = finPrint.FingerPrint(audio, r)
audio_fp.generate_fingerprint()

snapshot, r = finPrint.read_audio(path + snapshot_file + file_type)
snapshot_fp = finPrint.FingerPrint(snapshot, r)
snapshot_fp.generate_fingerprint()

# print("Average matches for Snapshot: ", basic_search(audio_fp, snapshot_fp))
# shrinking_search(song_fp=audio_fp, sample_fp=snapshot_fp, start_time=0, center_time=7)

sliding_lengths = np.arange(1, 2.0, 0.33)
windows_coll = []
matches_coll = []
hits_coll = []

for i in sliding_lengths:
    file_name = path + audio_file + '_sample_{}_'.format(i) + file_type
    _, windows, matches, hits = autowindow_search(song_fp=audio_fp,
                                                  window_length=i,
                                                  write_name=None,
                                                  to_plot=False,
                                                  get_info=True)
    matches = matches / i
    hits = hits / i
    windows_coll.append(windows)
    matches_coll.append(matches)
    hits_coll.append(hits)

plt.figure()
plt.title("Average Matches per Window using Different Sliding Window Lengths")
plt.xlabel("Window Starting Point (s)")
plt.ylabel("Average Number of Matches")
count = 0
for i in range(0, len(windows_coll)):
    plt.plot(windows_coll[i], matches_coll[i], 'C{}-'.format(i), label=str(sliding_lengths[i]))
plt.legend()
plt.show(block=False)

plt.figure()
plt.title("Matches per pair using Different Sliding Window Lengths")
plt.xlabel("Pair Start Time (s)")
plt.ylabel("Number of matches in database")
count = 0
for i in range(0, len(hits_coll)):
    plt.plot(audio_fp.pairs[:, finPrint.PAIR_TIME_IDX], hits_coll[i], 'C{}x'.format(i), label=str(sliding_lengths[i]))
plt.legend()
plt.show()
