import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io.wavfile import read
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure,
                                      iterate_structure, binary_erosion)
from operator import itemgetter

######################################################################
IDX_FREQ = 0
IDX_TIME = 2
IDX_TDELTA = 4
######################################################################


# Read audio as a time-signal
def read_audio(filename):
    fs, sound = read(filename, mmap=False)
    return sound, fs


# Transform audio to Time-Frequency Domain with STFT
def transform_stft(sound, fs, seg_len=1000):
    f, t, zxx = signal.stft(sound, fs, nperseg=seg_len)
    return f, t, zxx


def visualize_stft(f, t, zxx):
    plt.figure()
    plt.pcolormesh(t, f, np.abs(zxx), vmin=0, vmax=5)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)


def plot_peaks(f, t, peaks, threshold=-1):
    y, x = np.where(peaks != 0)
    plt.figure()
    plt.plot(t[x], f[y], 'rx')
    if (threshold >= 0):
        title = 'STFT Peaks, Threshold = {}'.format(threshold)
    else:
        title = 'STFT Peaks'
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)


def plot_pairs(f, t, peaks, pairs, inc=50):
    plot_peaks(f, t, peaks)
    count = 0
    f_cur = 0
    t_cur = 0

    for k in range(0, pairs.shape[0]):
        if (f_cur != pairs[k, IDX_FREQ]) or (t_cur != pairs[k, IDX_TIME]):
            f_cur = pairs[k, IDX_FREQ]
            t_cur = pairs[k, IDX_TIME]
            count += 1

        if count % inc == 0:
            plt.plot([pairs[k, IDX_TIME], pairs[k, IDX_TIME + 1]],
                     [pairs[k, IDX_FREQ], pairs[k, IDX_FREQ + 1]], 'b-')
            plt.plot(pairs[k, IDX_TIME], pairs[k, IDX_FREQ], 'ko')
            plt.plot(pairs[k, IDX_TIME + 1], pairs[k, IDX_FREQ + 1], 'b.')

    plt.title('Peak Pairs')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)
    return 0


# Find the index of the value closest to a0 in a
def find_nearest_index(a, a0):
    "Index in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin(axis=None)
    idx_nd = np.unravel_index(np.abs(a - a0).argmin(axis=None), a.shape)
    return idx_nd


def find_peaks_shift(f, t, zxx, threshold=1.0):
    spect = np.abs(zxx)

    # Find all peaks in the spectrogram
    peaks = np.ones(spect.shape)
    for h in range(-1, 2):
        for v in range(-1, 2):
            if h == 0 & v == 0:
                continue
            pzxx_shift = np.roll(spect, shift=[h, v], axis=[0, 1])
            peaks_new = (spect > pzxx_shift)
            peaks_rebuilt = (peaks == peaks_new)
            peaks = peaks_rebuilt

    # Prune by amplitude
    spect_peaks = np.copy(spect)
    spect_peaks[peaks != True] = 0

    # Coarse Segmentation: Split into equally spaced zones
    fMax = f.shape[0]
    tMax = t.shape[0]

    # Aim for 2-second, 5000 Hz windows
    fInc = find_nearest_index(f, 1000)[0]
    tInc = find_nearest_index(t, 1)[0]

    for sf in range(0, fMax, fInc):
        sf_next = sf + fInc
        if (sf_next > fMax):
            sf_next = fMax

        for st in range(0, tMax, tInc):
            st_next = st + tInc
            if st_next > tMax:
                st_next = tMax

            this_thresh = threshold
            if f[sf] < 500:
                this_thresh = this_thresh * 1.5
            if f[sf] > 3000:
                this_thresh = this_thresh * 1.5
            elif f[sf] > 10000:
                this_thresh = this_thresh * 2.0
            elif f[sf] > 15000:
                this_thresh = this_thresh * 2.5
            if this_thresh > 1.0:
                this_thresh = 1.0

            spect_window = spect_peaks[sf: sf_next, st: st_next]
            peaks_thresh = np.max(spect_window) * this_thresh

            spect_window[spect_window < peaks_thresh] = 0

    num_peaks = np.sum(spect_peaks != 0)
    return spect_peaks, num_peaks


# Anchor Point Pairing
# Pair peaks with each other
def pair_peaks(f, t, peaks, fanout=3):
    pairs = np.zeros((0, 5))
    fidx, tidx = np.where(peaks != 0)

    # Create the target zone
    freq_tz_max = 1000
    time_tz_max = 100

    for i in range(0, fidx.size):
        freq_tz_idx = np.where(
            np.logical_and(fidx >= fidx[i] - freq_tz_max, fidx <= fidx[i] + freq_tz_max))  # Frequency target zone
        time_tz_idx = np.where(
            np.logical_and(tidx >= tidx[i] + 1, tidx <= tidx[i] + time_tz_max))  # Time target zone
        # target_idx = np.intersect1d(freq_tz_idx, time_tz_idx)
        target_idx = np.asarray(time_tz_idx, int).T
        # Pair a fixed number of peaks within the target zone
        max_pairs = min(target_idx.shape[0], fanout)
        for j in range(0, max_pairs):
            f1 = f[fidx[i]]
            f2 = f[fidx[target_idx[j]]]
            t1 = t[tidx[i]]
            t2 = t[tidx[target_idx[j]]]
            t_delta = np.abs(t1 - t2)
            pairs = np.vstack((pairs, np.array([f1, f2, t1, t2, t_delta])))

    pairs = pairs.astype(int)
    pairs.sort(axis=0)
    num_pairs = pairs.shape[0]
    return pairs, num_pairs

# Create a 2-second sliding window

# Query the pair table
def search_for_pair(pairs, query):

    # Create the target zone
    ftz = 1000
    ttz = 0.5
    num_pairs = pairs.shape[0]
    matches = []
    for i in range(0, num_pairs):
        if pairs[i][IDX_FREQ] != query[IDX_FREQ]:
            continue
        elif pairs[i][IDX_FREQ + 1] != query[IDX_FREQ + 1]:
            continue
        elif pairs[i][IDX_TDELTA] != query[IDX_TDELTA]:
            continue
        else:
            matches.append(i)
    return np.asarray(matches, int)


# Record results of hash table fingerprints
def accumulate_searches(pairs, query_pairs):
    pair_matches = np.zeros(pairs.shape[0])
    for query in query_pairs:
        matches = search_for_pair(pairs, query)
        pair_matches[matches] += 1
    return pair_matches


# Timestamp extraction
def merge_samples(pair_matches):
    pass


for i in range(0, 1):
    sound, r = read_audio('./main/bin/t{}.wav'.format(i + 1))
    f, t, zxx = transform_stft(sound, r, 10000)
    visualize_stft(f, t, zxx)
    # kRange = np.linspace(0, 1, 10)
    # for k in kRange:
    #     peaks, num_peaks = find_peaks_shift(f, t, Zxx, k)
    #     print(num_peaks)
    #     visualize_stft(f, t, peaks)
    #     visualize_peaks(f, t, peaks, k)
    peaks, num_peaks = find_peaks_shift(f, t, zxx, 1.0)
    print("Number of peaks: {}".format(num_peaks))
    plot_peaks(f, t, peaks)
    pairs, num_pairs = pair_peaks(f, t, peaks)
    plot_pairs(f, t, peaks, pairs)
    print("Number of peak pairs: {}".format(num_pairs))

    # Create a small sound clip
    tIdx = find_nearest_index(t, 2)[0]
    t_clip = t[0:tIdx]
    zxx_clip = zxx[:, 0:tIdx]
    visualize_stft(f, t_clip, zxx_clip)
    peaks_clip, num_peaks_clip = find_peaks_shift(f, t_clip, zxx_clip, 1.0)
    pairs_clip, num_pairs_clip = pair_peaks(f, t_clip, peaks_clip)
    plot_pairs(f, t, peaks, pairs, inc=1)

    matches = search_for_pair(pairs, pairs_clip[0, :])
    print(pairs_clip[0, :])
    print(pairs[0:50])
    print(matches)
plt.show()
