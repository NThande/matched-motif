import hashlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io.wavfile import read
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure,
                                      iterate_structure, binary_erosion)

# Lifted from Dejavu Fingerprinting System
IDX_FREQ_I = 0
IDX_TIME_J = 1

######################################################################
# Sampling rate, related to the Nyquist conditions, which affects
# the range frequencies we can detect.
DEFAULT_FS = 44100

######################################################################
# Size of the FFT window, affects frequency granularity
DEFAULT_WINDOW_SIZE = 4096

######################################################################
# Ratio by which each sequential window overlaps the last and the
# next window. Higher overlap will allow a higher granularity of offset
# matching, but potentially more fingerprints.
DEFAULT_OVERLAP_RATIO = 0.5

######################################################################
# Degree to which a fingerprint can be paired with its neighbors --
# higher will cause more fingerprints, but potentially better accuracy.
DEFAULT_FAN_VALUE = 15

######################################################################
# Minimum amplitude in spectrogram in order to be considered a peak.
# This can be raised to reduce number of fingerprints, but can negatively
# affect accuracy.
DEFAULT_AMP_MIN = 10

######################################################################
# Number of cells around an amplitude peak in the spectrogram in order
# for Dejavu to consider it a spectral peak. Higher values mean less
# fingerprints and faster matching, but can potentially affect accuracy.
PEAK_NEIGHBORHOOD_SIZE = 20

######################################################################
# Thresholds on how close or far fingerprints can be in time in order
# to be paired as a fingerprint. If your max is too low, higher values of
# DEFAULT_FAN_VALUE may not perform as expected.
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

######################################################################
# If True, will sort peaks temporally for fingerprinting;
# not sorting will cut down number of fingerprints, but potentially
# affect performance.
PEAK_SORT = True

def fingerprint(channel_samples, Fs=DEFAULT_FS,
                wsize=DEFAULT_WINDOW_SIZE,
                wratio=DEFAULT_OVERLAP_RATIO,
                fan_value=DEFAULT_FAN_VALUE,
                amp_min=DEFAULT_AMP_MIN):
    """
    FFT the channel, log transform output, find local maxima, then return
    locally sensitive hashes.
    """
    # FFT the signal and extract frequency components
    spect = plt.mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=Fs,
        window=plt.mlab.window_hanning,
        noverlap=int(wsize * wratio))[0]

    # apply log transform since specgram() returns linear array
    spect = 10 * np.log10(spect)
    spect[spect == -np.inf] = 0  # replace infs with zeros

    # find local maxima
    local_maxima = get_2D_peaks(spect, plot=False, amp_min=amp_min)

    # return hashes
    return generate_hashes(local_maxima, fan_value=fan_value)

def generate_hashes(peaks, fan_value=DEFAULT_FAN_VALUE):
    """
    Hash list structure:
       sha1_hash[0:20]    time_offset
    [(e05b341a9b77a51fd26, 32), ... ]
    """
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):

                freq1 = peaks[i][IDX_FREQ_I]
                freq2 = peaks[i + j][IDX_FREQ_I]
                t1 = peaks[i][IDX_TIME_J]
                t2 = peaks[i + j][IDX_TIME_J]
                t_delta = t2 - t1

                if t_delta >= MIN_HASH_TIME_DELTA and t_delta <= MAX_HASH_TIME_DELTA:
                    h = hashlib.sha1(
                        "%s|%s|%s" % (str(freq1), str(freq2), str(t_delta)))
                    yield (h.hexdigest()[0:FINGERPRINT_REDUCTION], t1)
# End citation


# Read audio as a time-signal
def read_audio(filename):
    fs, sound = read(filename, mmap=False)
    return sound, fs


# Transform audio to Time-Frequency Domain with STFT
def transform_stft(sound, fs, seg_len=1000):
    f, t, zxx = signal.stft(sound, fs, nperseg=seg_len)
    zxx = 10 * np.log10(zxx)
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


def plot_peak_pairs(f, t, peaks, pairs, inc=50):
    plot_peaks(f, t, peaks)
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

# Lifted from Dejavu fingerprinting system
def get_2D_peaks(spect, plot=False, amp_min=DEFAULT_AMP_MIN):
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphology.iterate_structure.html#scipy.ndimage.morphology.iterate_structure
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our fliter shape
    local_max = maximum_filter(spect, footprint=neighborhood) == spect
    background = (spect == 0)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    # Boolean mask of arr2D with True at peaks
    detected_peaks = local_max - eroded_background

    # extract peaks
    amps = spect[detected_peaks]
    j, i = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = [x for x in peaks if x[2] > amp_min]  # freq, time, amp

    # get indices for frequency and time
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]

    if plot:
        # scatter of the peaks
        fig, ax = plt.subplots()
        ax.imshow(spect)
        ax.scatter(time_idx, frequency_idx)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()

    return zip(frequency_idx, time_idx)

# Anchor Point Pairing
# Pair peaks with each other
def get_peak_pairs(peaks, fanout=3):
    pairs = np.zeros((0, 4))
    fidx, tidx = np.where(peaks != 0)

    # Create the target zone
    ftz_max = 1000
    ttz_max = 100

    for i in range(0, fidx.size):
        ftz_idx = np.where(
            np.logical_and(fidx >= fidx[i] - ftz_max, fidx <= fidx[i] + ftz_max))  # Frequency target zone
        ttz_idx = np.where(
            np.logical_and(tidx >= tidx[i] + 1, tidx <= tidx[i] + ttz_max))  # Time target zone
        tzone_idx = np.intersect1d(ftz_idx, ttz_idx)

        # Pair a fixed number of peaks within the target zone
        max_pairs = min(tzone_idx.size, fanout)
        for j in range(0, max_pairs):
            pairs = np.vstack((pairs, np.array([fidx[i], fidx[tzone_idx[j]], tidx[i], tidx[tzone_idx][j]])))

    pairs = pairs.astype(int)
    num_pairs = pairs.shape[0]
    return pairs, num_pairs


# Anchor Point Hashing
# For each pair:
# Create a hash for each pair
# Move hashes into database
def convert_to_hashes(pairs):
    hashtable = {}
    for pV in pairs:
        f_diff = pV[0] - pV[1]
        t_diff = pV[2] - pV[3]
        time_stamp = pV[2]
        hash = np.round(np.abs(f_diff) * 1000000) + np.round(np.abs(t_diff) * 1000) + pV[0]
        hashtable[hash] = time_stamp
        print(hash)
    return hashtable


# Create a sliding window
# Query the hash table
# Record results of hash table fingerprints
# Timestamp extraction

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
    pairs, num_pairs = get_peak_pairs(peaks)
    plot_peak_pairs(f, t, peaks, pairs)
    print("Number of peak pairs: {}".format(num_pairs))
    pair_table = convert_to_hashes(pairs)
    print("Number of unique hashes: {}".format(len(pair_table)))
plt.show()
