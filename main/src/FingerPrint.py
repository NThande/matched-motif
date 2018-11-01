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


# Peak Identification
def find_peaks_shift(f, t, Zxx):
    pzxx = np.abs(Zxx)
    threshold = 100

    # Find all peaks in the zone
    peaks = np.ones(pzxx.shape)
    for h in range(-1, 2):
        for v in range(-1, 2):
            pzxx_shift = np.roll(pzxx, shift=[h, v], axis=[0, 1])
            peaks_new = (pzxx > pzxx_shift)
            peaks_rebuilt = (peaks == peaks_new)
            peaks = peaks_rebuilt

    # Prune by amplitude
    pzxx_peaks = np.copy(pzxx)
    pzxx_peaks[peaks != True] = 0
    pzxx_peaks[np.where(pzxx_peaks < threshold)] = 0

    # Merge Peaks from zones into a single vector
    visualize_stft(f, t, peaks)
    visualize_stft(f, t, pzxx_peaks)

    return pzxx_peaks


def find_peaks_filt(f, t, Zxx):
    pzxx = np.abs(Zxx)
    threshold = 100
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

for i in range(0, 10):
    sound, r = read_audio('./main/bin/t{}.wav'.format(i + 1))
    f, t, Zxx = transform_stft(sound, r, 10000)
    print(Zxx.shape)
    print(f.shape)
    print(t.shape)
    #visualize_stft(f, t, Zxx)
#plt.show()
sound, r = read_audio('./main/bin/t1.wav')
f, t, Zxx = transform_stft(sound, r, 1000)
find_peaks_shift(f, t, Zxx)
plt.show()