import numpy as np
from scipy import signal
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters


# Read audio as a time-signal
def read_audio(filename):
    fs, sound = read(filename, mmap=False)
    return sound, fs


# Transform audio to Time-Frequency Domain
# Short = Time Fourier Transform
# Stride: Half-Windows
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

def visualize_peaks(f, t, x, y):
    plt.figure()
    plt.plot(t[x], f[y], 'rx')
    plt.title('STFT Peaks')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)


# Coarse segmentation for peak identification
def segment_coarse(f, t, Zxx):
    f_zone = 3
    t_zone = 3
    f_len = f.shape[0]
    t_len = t.shape[0]
    for freq in range(0, f_zone, f_len):
        for time in range(0, t_zone, t.shape[0]):
            f_end = freq + f_zone if freq + f_zone < f_len else f_len
            t_end = time + t_zone if time + t_zone < t_len else t_len
            peaks = find_peaks_shift(f[freq: f_end], t[time: t_end],
                                     Zxx[freq: f_end, time: t_end])

    return None


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

    # Time-Frequency segmentation
    fMax = f.shape[0]
    tMax = t.shape[0]
    # Aim for 2-second, 5000 Hz windows
    fInc = find_nearest_index(f, 2000)[0]
    tInc = find_nearest_index(t, 1.5)[0]

    for sf in range(0, fMax, fInc):
        sfNext = sf + fInc
        if (sfNext > fMax):
            sfNext = fMax

        for st in range (0, tMax, tInc):
            stNext = st + tInc
            if (stNext > tMax):
                stNext = tMax

            thisThresh = threshold
            if (f[sf] > 3000):
                thisThresh = thisThresh * 1.5
            elif (f[sf] > 10000):
                thisThresh = thisThresh * 2.0
            elif (f[sf] > 15000):
                thisThresh = thisThresh * 2.5
            if (thisThresh > 1.0):
                thisThresh = 1.0

            spect_window = spect_peaks[sf : sfNext, st: stNext]
            peaks_thresh = np.max(spect_window) * thisThresh

            spect_window[spect_window < peaks_thresh] = 0
            # print(np.sum(spect_peaks != 0))

    # Prune by amplitude
   # spect_peaks = np.copy(spect)
    #spect_peaks[peaks != True] = 0

    #peaks_thresh = np.max(spect_peaks) * threshold
   # spect_peaks[spect_peaks < peaks_thresh] = 0
    print(np.sum(spect_peaks != 0))

    y, x = np.where(spect_peaks != 0)
    visualize_stft(f, t, spect_peaks)
    visualize_peaks(f, t, x, y)

    return spect_peaks


def find_peaks_filt(f, t, Zxx, threshold=100):
    pzxx = np.abs(Zxx)
    neighborhood_size = [5, 1]

    # Find all peaks in the zone
    data_max = filters.maximum_filter(pzxx, neighborhood_size)
    maxima = (pzxx == data_max)
    data_min = filters.minimum_filter(pzxx, neighborhood_size)

    # Prune by amplitude
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    visualize_stft(f, t, maxima)

    # Merge Peaks from zones into a single vector
    labeled, num_objects = ndimage.label(maxima)
    print(num_objects)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y.append(y_center)

    plt.figure()
    plt.autoscale(True)
    plt.plot(x, y, 'rx')

    return x, y


# Coarse Segmentation: Split into equally spaced zones
# Anchor Point Hashing
# For each peak:
# Set the target zone
# Pair a fixed number of peaks within the target zone
# Create a hash for each pair
# Merge hashes into a single database

# Create a sliding window
# Query the hash table
# Record results of hash table fingerprints
# Timestamp extraction

for i in range(0, 1):
    sound, r = read_audio('./main/bin/t{}.wav'.format(i + 1))
    f, t, Zxx = transform_stft(sound, r, 10000)
    visualize_stft(f, t, Zxx)
    find_peaks_shift(f, t, Zxx)
    # find_peaks_filt(f, t, Zxx)
plt.show()
