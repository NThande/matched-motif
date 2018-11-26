import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io.wavfile import read
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure,
                                      iterate_structure, binary_erosion)

# Modified from Dejavu Fingerprinting System (as of 11.21.18)
IDX_FREQ = 0
IDX_TIME = 1
DEFAULT_FS = 44100
DEFAULT_WINDOW_SIZE = 4096
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_FAN_VALUE = 15
DEFAULT_AMP_MIN = 10
PEAK_NEIGHBORHOOD_SIZE = 20
MIN_TIME_DELTA = 0
MAX_TIME_DELTA = 200
PEAK_SORT = True


# FFT the channel, log transform output, find local maxima, then return
# locally sensitive hashes.
def fingerprint(samples, Fs=DEFAULT_FS,
                wsize=DEFAULT_WINDOW_SIZE,
                wratio=DEFAULT_OVERLAP_RATIO,
                fan_value=DEFAULT_FAN_VALUE,
                amp_min=DEFAULT_AMP_MIN):

    # FFT the signal and extract frequency components
    freqs, times, spect = signal.stft(
        samples,
        nfft=wsize,
        fs=Fs,
        window='hann',
        nperseg=int(wsize),
        noverlap=int(wsize * wratio))

    # spect, freqs, times = plt.mlab.specgram(
    #     samples,
    #     NFFT=wsize,
    #     Fs=Fs,
    #     window=plt.mlab.window_hanning,
    #     noverlap=int(wsize * wratio))

    # apply log transform since specgram() returns linear array
    spect[spect == -np.inf] = 0
    spect = np.abs(spect)

    # find local maxima
    peaks = get_2D_peaks(spect, amp_min=amp_min)
    # plt.figure()
    # plt.plot(times[peaks[:, 1]], freqs[peaks[:, 0]], 'rx')
    pairs = generate_pairs(peaks, fan_value)
    # plt.figure()
    # plt.plot([times[pairs[:, 3]], times[pairs[:, 4]]],
    #          [freqs[pairs[:, 0]], freqs[pairs[:, 1]]], 'b-')
    # plt.plot(times[pairs[:, 3]], freqs[pairs[:, 0]], 'kx')
    # plt.plot(times[pairs[:, 4]], freqs[pairs[:, 1]], 'k*')
    return pairs


def get_2D_peaks(spect, amp_min=DEFAULT_AMP_MIN):
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphology.iterate_structure.html#scipy.ndimage.morphology.iterate_structure
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our fliter shape
    local_max = maximum_filter(spect, footprint=neighborhood) == spect
    background = (spect == 0)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    # Boolean mask of arr2D with True at peaks
    detected_peaks = local_max ^ eroded_background

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

    peaks = np.zeros([len(frequency_idx), 2])
    peaks[:, 0] = frequency_idx
    peaks[:, 1] = time_idx
    return peaks.astype(int)


def generate_pairs(peaks, fan_value=DEFAULT_FAN_VALUE):

    if PEAK_SORT:
        peaks.sort(axis=0)
    pairs = np.zeros((0, 5))

    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):

                freq1 = peaks[i, IDX_FREQ]
                freq2 = peaks[i + j, IDX_FREQ]
                t1 = peaks[i, IDX_TIME]
                t2 = peaks[i + j, IDX_TIME]
                t_delta = t2 - t1

                if MIN_TIME_DELTA <= t_delta <= MAX_TIME_DELTA:
                    pairs = np.vstack((pairs, np.array([freq1, freq2, t_delta, t1, t2])))

    print(pairs.shape)
    return pairs.astype(int)


# Read audio as a time-signal
def read_audio(filename):
    fs, sound = read(filename, mmap=False)
    return sound, fs



# Query the pair table
def search_for_pair(pairs, query):
    return np.asarray(np.where(pairs == query), int)
    # # Create the target zone
    # ftz = 1000
    # ttz = 0.5
    # num_pairs = pairs.shape[0]
    # matches = []
    # for i in range(0, num_pairs):
    #     if pairs[i][IDX_FREQ] != query[IDX_FREQ]:
    #         continue
    #     elif pairs[i][IDX_FREQ + 1] != query[IDX_FREQ + 1]:
    #         continue
    #     elif pairs[i][IDX_TDELTA] != query[IDX_TDELTA]:
    #         continue
    #     else:
    #         matches.append(i)
    # return np.asarray(matches, int)

# Test basic Shazam-style identification
# sample_length = 2
# for i in range(3, 4):
#     sound, r = read_audio('./main/bin/t{}.wav'.format(i + 1))
#     sound_data = fingerprint(sound, r)
#     sample = sound[0:(r * sample_length)]
#     sample_data = fingerprint(sample, r)
#     print(sample_data.shape)
#     print(sound_data.shape)
#     print(sample_data[0])
#     sample_matches = np.zeros([sample_data.shape[0], 2])
#     sample_matches[:, 0] = np.arange(0, sample_data.shape[0])
#     for sample_pair in sample_data:
#         matches = search_for_pair(sound_data, sample_pair)
#         print(matches.shape[1])
# plt.show()

