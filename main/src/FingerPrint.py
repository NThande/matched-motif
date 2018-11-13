import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io.wavfile import read


# Read audio as a time-signal
def read_audio(filename):
    fs, sound = read(filename, mmap=False)
    return sound, fs


# Transform audio to Time-Frequency Domain with STFT
def transform_stft(sound, fs, segLen=1000):
    f, t, Zxx = signal.stft(sound, fs, nperseg=segLen)
    return f, t, Zxx


def visualize_stft(f, t, Zxx):
    plt.figure()
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=5)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)


def visualize_peaks(f, t, peaks, threshold=-1):
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


def visualize_peak_pairs(f, t, peaks, pairs, inc=50):

    visualize_peaks(f, t, peaks)
    count = 0
    f_cur = 0
    t_cur = 0

    for k in range(0, pairs.shape[0]):
        if (f_cur != pairs[k, 0]) or (t_cur != pairs[k, 2]):
            f_cur = pairs[k, 0]
            t_cur = pairs[k, 2]
            count += 1

        if count % inc == 0:
            plt.plot([t[pairs[k, 2]], t[pairs[k, 3]]], [f[pairs[k, 0]], f[pairs[k, 1]]], 'b-')
            plt.plot(t[pairs[k, 3]], f[pairs[k, 1]], 'b.')
            plt.plot(t[pairs[k, 2]], f[pairs[k, 0]], 'ko')

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


# Peak Identification
def find_peaks_shift(f, t, Zxx, threshold=1.0):
    spect = np.abs(Zxx)

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
        sfNext = sf + fInc
        if (sfNext > fMax):
            sfNext = fMax

        for st in range(0, tMax, tInc):
            stNext = st + tInc
            if (stNext > tMax):
                stNext = tMax

            thisThresh = threshold
            if (f[sf] < 500):
                thisThresh = thisThresh * 1.5
            if (f[sf] > 3000):
                thisThresh = thisThresh * 1.5
            elif (f[sf] > 10000):
                thisThresh = thisThresh * 2.0
            elif (f[sf] > 15000):
                thisThresh = thisThresh * 2.5
            if (thisThresh > 1.0):
                thisThresh = 1.0

            spect_window = spect_peaks[sf: sfNext, st: stNext]
            peaks_thresh = np.max(spect_window) * thisThresh

            spect_window[spect_window < peaks_thresh] = 0

    num_peaks = np.sum(spect_peaks != 0)
    return spect_peaks, num_peaks


# Anchor Point Pairing
# Pair peaks with each other
def pair_peaks(peaks, fanout=6):
    pairs = np.zeros((0, 4))
    fidx, tidx = np.where(peaks != 0)

    # Create the target zone
    ftz_max = 1000
    ttz_max = 100

    plt.figure()
    plt.plot(tidx, fidx, 'rx')
    for i in range(0, fidx.size):
        ftz_idx = np.where(
            np.logical_and(fidx >= fidx[i] - ftz_max, fidx <= fidx[i] + ftz_max))  # Frequency target zone
        ttz_idx = np.where(np.logical_and(tidx >= tidx[i] + 1, tidx <= tidx[i] + ttz_max))  # Frequency target zone
        tzone_idx = np.intersect1d(ftz_idx, ttz_idx)

        # Pair a fixed number of peaks within the target zone
        max_pairs = min(tzone_idx.size, fanout)
        for j in range(0, max_pairs):
            pairs = np.vstack((pairs, np.array([fidx[i], fidx[tzone_idx[j]], tidx[i], tidx[tzone_idx][j]])))
            if i % 50 == 0:
                plt.plot([tidx[i], tidx[tzone_idx[j]]], [fidx[i], fidx[tzone_idx][j]], 'b-')
                plt.plot(tidx[tzone_idx[j]], fidx[tzone_idx][j], 'b.')
                plt.plot(tidx[i], fidx[i], 'ko')
    pairs = pairs.astype(int)
    num_pairs = pairs.shape[0]
    return pairs, num_pairs


# Anchor Point Hashing
# For each pair:
# Create a hash for each pair
# Merge hashes into a single database
def convert_to_hashes(pairs):
    return 0


# Create a sliding window
# Query the hash table
# Record results of hash table fingerprints
# Timestamp extraction

for i in range(0, 1):
    sound, r = read_audio('./main/bin/t{}.wav'.format(i + 1))
    f, t, Zxx = transform_stft(sound, r, 10000)
    visualize_stft(f, t, Zxx)
    # kRange = np.linspace(0, 1, 10)
    # for k in kRange:
    #     peaks, num_peaks = find_peaks_shift(f, t, Zxx, k)
    #     print(num_peaks)
    #     visualize_stft(f, t, peaks)
    #     visualize_peaks(f, t, peaks, k)
    peaks, num_peaks = find_peaks_shift(f, t, Zxx, 1.0)
    print(num_peaks)
    visualize_peaks(f, t, peaks)
    pairs, num_pairs = pair_peaks(peaks)
    visualize_peak_pairs(f, t, peaks, pairs)
    print(num_pairs)
plt.show()
