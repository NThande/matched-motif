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
    snip_end = 2*center_time - start_time
    if (snip_end > sound.shape[0] / r):
        snip_end = sound.shape[0] / r

    # snip_matches = np.zeros(6)
    # snip_lengths = np.arange(13, 2, -2)
    # snip_lengths = np.arange(snip_end - snip_start )
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
        print(snip_lengths)
        print(snip_matches)
        plt.plot(snip_lengths, snip_matches, 'rx-')
        plt.xlabel("Snippet Length")
        plt.ylabel("Number of matches in database")
        plt.title("Number of matches for fixed snippet vs length of overall fingerprint track")

    return samp_hits


# Apply a sliding window of a part of the song to the rest of the song
def autowindow_search(song_fp, to_write=True, to_plot=True):
    sound = song_fp.sound
    r = song_fp.fs
    song_length = np.ceil(sound.shape[0] / r)
    window_length = 2
    snap_num = int(song_length - window_length)
    snap_windows = np.arange(0, snap_num)
    snap_matches = np.zeros(snap_num)
    snap_hits = np.zeros(song_fp.pairs.shape[0])
    for i in range(0, snap_num):
        snap_start = i * r
        snap_end = (i + window_length) * r
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
    max_sample = sound[max_samp_num * r + 1: (max_samp_num + window_length) * r]

    if to_plot:
        plt.figure()
        plt.plot(snap_windows, snap_matches, 'rx-')
        plt.xlabel("Snapshot Starting Point (s)")
        plt.ylabel("Number of matches in database")
        plt.title("Average number of matches for Fixed Length Sliding Windows")

        plt.figure()
        plt.plot(song_fp.pairs[:, finPrint.PAIR_TIME_IDX], snap_hits, 'rx', label='Pair Start Time')
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Number of matches")
        plt.title("Matches per pair using Fixed Length Sliding Window")
        plt.show()

    if to_write:
        write('./main/bin/unique/test.wav', r, max_sample)

    return max_sample


audio, r = finPrint.read_audio('./main/bin/unique/hello_train.wav')
audio_fp = finPrint.FingerPrint(audio, r)
audio_fp.generate_fingerprint()

snapshot, r = finPrint.read_audio('./main/bin/unique/hello_test.wav')
snapshot_fp = finPrint.FingerPrint(snapshot, r)
snapshot_fp.generate_fingerprint()

print("Average matches for Snapshot: ", basic_search(audio_fp, snapshot_fp))
shrinking_search(song_fp=audio_fp, sample_fp=snapshot_fp, start_time=0, center_time=7)
autowindow_search(song_fp=audio_fp)

# print(sample_pairs[0].astype(int))
# match_vect = np.zeros(sample_pairs.shape[0])
# for i in range(0, sample_pairs.shape[0]):
#     _, match_vect[i] = sound_fp.search_for_pair(sample_pairs[i])
# matches = np.average(match_vect)

# # Apply a sliding window on the sound itself
# sound_length = np.ceil(sound.shape[0] / r)
# snap_length = 2
# snap_num = int(sound_length - snap_length)
# snap_windows = np.arange(0, snap_num)
# snap_matches = np.zeros(snap_num)
# snap_hits = np.zeros(sound_pairs.shape[0])
# for i in range(0, snap_num):
#     snap_start = i * r
#     snap_end = (i + snap_length) * r
#     if snap_end > sound.shape[0]:
#         snap_end = sound.shape[0]
#     snap = sound[snap_start: snap_end]
#     snap_peaks, snap_data = finPrint.fingerprint(snap, Fs=r)
#     match_vect = np.zeros(snap_data.shape[0])
#     for j in range(0, snap_data.shape[0]):
#         match_idx, match_vect[j] = finPrint.search_for_pair(sound_pairs, snap_data[j])
#         snap_hits[match_idx] += 1
#     snap_matches[i] = np.average(match_vect)

# print(snap_hits)
# plt.figure()
# print(snap_matches)
# plt.plot(snap_windows, snap_matches, 'rx-')
# plt.xlabel("Snapshot Starting Point (s)")
# plt.ylabel("Number of matches in database")
# plt.title("Average number of matches for Fixed Length Sliding Windows")
#
# max_samp_idx = np.argmax(snap_matches)
# max_samp_num = snap_windows[max_samp_idx]
# max_sample = sound[max_samp_num * r + 1: (max_samp_num + snap_length) * r]
# write('./main/bin/unique/test.wav', r, max_sample)
#
# plt.figure()
# plt.plot(sound_pairs[:, PAIR_TIME_IDX], snap_hits, 'rx', label='Pair Start Time')
# plt.legend()
# plt.xlabel("Time (s)")
# plt.ylabel("Number of matches")
# plt.title("Matches per pair using Fixed Length Sliding Window")
# plt.show()

# snip_end = 13
# snip_matches = np.zeros(6)
# snip_lengths = np.arange(13, 2, -2)
# samp_hits = np.zeros(sound_pairs.shape[0])
# for i in range(0, snip_matches.shape[0] - 1):
#     snippet_peaks, snippet_data = finPrint.fingerprint(sound[i * r: snip_end * r], Fs=r)
#     match_vect = np.zeros(sample_pairs.shape[0])
#     for j in range(0, sample_pairs.shape[0]):
#         match_idx, match_vect[j] = finPrint.search_for_pair(snippet_data, sample_pairs[j])
#         samp_hits[match_idx] += 1
#     snip_end = snip_end - 1
#     snip_matches[i] = np.average(match_vect)
# plt.figure()
# print(snip_lengths)
# print(snip_matches)
# plt.plot(snip_lengths, snip_matches, 'rx-')
# plt.xlabel("Snippet Length")
# plt.ylabel("Number of matches in database")
# plt.title("Number of matches for fixed snippet vs length of overall fingerprint track")
#
# plt.figure()
# plt.plot(sound_pairs[:, finPrint.PAIR_TIME_IDX], samp_hits, 'rx', label='Pair End Time')
# plt.legend()
# plt.xlabel("Time (s)")
# plt.ylabel("Number of matches")
# plt.title("Matches per pair using Fixed Sample")
#

# plt.show()