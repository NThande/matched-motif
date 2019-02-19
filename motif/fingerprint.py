import librosa
import matplotlib.pyplot as plt
import numpy as np
from librosa import display
from scipy import signal
from scipy.io.wavfile import read
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure,
                                      iterate_structure, binary_erosion)

import fileutils

# Modified from Dejavu Fingerprinting System (as of 11/21/18)
FREQ_IDX = 0
PEAK_TIME_IDX = 1
PAIR_TIME_IDX = 2
PAIR_TDELTA_IDX = 4
DEFAULT_FS = 44100
DEFAULT_WINDOW_SIZE = 4096
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_FAN_VALUE = 15
DEFAULT_AMP_MIN = 10
PEAK_NEIGHBORHOOD_SIZE = 20
MIN_TIME_DELTA = 0
MAX_TIME_DELTA = 200


# def __init__(self, sound, fs=DEFAULT_FS):
#     self.sound = sound
#     self.peaks = None
#     self.pairs = None
#     self.fs = fs
#     self.hasFingerPrint = False


# FFT the channel, log transform output, find local maxima, then return
# locally sensitive hashes.
def fingerprint(audio, fs=DEFAULT_FS,
                wsize=DEFAULT_WINDOW_SIZE,
                wratio=DEFAULT_OVERLAP_RATIO,
                fan_value=DEFAULT_FAN_VALUE,
                amp_min=DEFAULT_AMP_MIN,
                to_plot=False):
    # self.hasFingerPrint = False

    # FFT the signal and extract frequency components
    frequencies, times, sxx = transform_stft(audio, fs, wsize, wratio)
    sxx = librosa.stft(audio,
                       n_fft=wsize,
                       win_length=wsize,
                       hop_length=int(wsize * wratio),
                       window='hann'
                       )
    # if to_plot:
    #     plot_stft(frequencies, times, spectrogram)

    sxx[sxx == -np.inf] = 0
    sxx = np.abs(sxx)

    # find local maxima
    peaks_f, peaks_t = get_2d_peaks(sxx, amp_min=amp_min)
    # peaks = np.zeros([len(peaks_f), 2])
    peaks = np.stack((peaks_f, peaks_t), axis=1)
    # peaks[:, 0] = frequencies[peaks_f]
    # peaks[:, 1] = times[peaks_t]
    # peaks[:, 0] = peaks_f
    # peaks[:, 1] = peaks_t
    # peaks = peaks
    # if to_plot:
    #     self.plot_peaks()
    pairs = generate_pairs(fan_value)
    # pairs = pairs
    # if to_plot:
    #     plot_pairs(plot_inc)
    # hasFingerPrint = True
    # if to_plot:
    #     plot_spect_peaks_pairs(frequencies, times, spectrogram, plot_inc)
    return peaks, pairs


# Generate peak-pairs based on locally-sensitive target zone
def generate_pairs(peaks, fan_value=DEFAULT_FAN_VALUE, tol=0):
    peaks = np.unique(peaks, axis=0)
    pairs = np.zeros((0, 5))

    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):

                freq1 = peaks[i, FREQ_IDX]
                freq2 = peaks[i + j, FREQ_IDX]
                t1 = peaks[i, PEAK_TIME_IDX]
                t2 = peaks[i + j, PEAK_TIME_IDX]
                t_delta = t2 - t1

                if MIN_TIME_DELTA - tol <= t_delta <= MAX_TIME_DELTA + tol:
                    pairs = np.vstack((pairs, np.array([freq1, freq2, t1, t2, t_delta])))

    # Return dummy entry
    if pairs.shape[0] == 0:
        print("No pairs found, only {} peaks".format(peaks.shape[0]))
        return np.zeros((1, 5))
    pairs = np.unique(pairs, axis=0)
    return pairs


# Query the pair table
def linear_search(pairs, query):
    t_delta_tol = 10
    t_delta_matches = search_col(pairs[:, PAIR_TDELTA_IDX], query[PAIR_TDELTA_IDX], t_delta_tol)
    t_pairs = pairs[t_delta_matches]

    f_tol = 50
    f1_matches = search_col(t_pairs[:, FREQ_IDX], query[FREQ_IDX], f_tol)
    f1_pairs = t_pairs[f1_matches]

    f2_matches = search_col(f1_pairs[:, FREQ_IDX + 1], query[FREQ_IDX + 1], f_tol)
    f2_pairs = f1_pairs[f2_matches]

    num_matches = f2_pairs.shape[0]
    tf2_idx = (t_delta_matches[f1_matches])[f2_matches]
    return tf2_idx, num_matches


# # Plot Spectrogram peaks
# def plot_peaks(peaks):
#     if self.peaks is None:
#         return
#     plt.figure()
#     plt.plot(self.peaks[:, PEAK_TIME_IDX], self.peaks[:, FREQ_IDX], 'rx')
#     plt.title('STFT Peaks')
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     plt.show(block=False)

# # Plot spectogram peaks and one pair from every inc pairs
# def plot_pairs(self, inc=50):
#     if self.pairs is None:
#         return
#     pair_mask = np.zeros(self.pairs.shape[0]).astype(int)
#     for i in range(0, self.pairs.shape[0]):
#         if i % inc == 0: pair_mask[i] = i
#     pruned = self.pairs[pair_mask, :]
#
#     plt.figure()
#     plt.plot(self.peaks[:, PEAK_TIME_IDX], self.peaks[:, FREQ_IDX], 'rx')
#     plt.plot([pruned[:, PAIR_TIME_IDX], pruned[:, PAIR_TIME_IDX + 1]],
#              [pruned[:, FREQ_IDX], pruned[:, FREQ_IDX + 1]], 'b-')
#
#     plt.plot(pruned[:, PAIR_TIME_IDX], pruned[:, FREQ_IDX], 'kx')
#     plt.plot(pruned[:, PAIR_TIME_IDX + 1], pruned[:, FREQ_IDX + 1], 'k*')
#     plt.title('Peak Pairs')
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     plt.show(block=False)

# # Plot everything on the same plot
# def plot_spect_peaks_pairs(self, freqs, times, spectrogram, inc=50):
#     plt.figure()
#     plt.pcolormesh(times, freqs, np.abs(spectrogram), vmin=0, vmax=5)
#     plt.title('Spectogram Magnitude with Peaks & Sample Pairs')
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#
#     if self.peaks is None:
#         return
#     plt.plot(self.peaks[:, PEAK_TIME_IDX], self.peaks[:, FREQ_IDX], 'rx', label='Peaks')
#
#     if self.pairs is None:
#         return
#     pair_mask = np.zeros(self.pairs.shape[0]).astype(int)
#     for i in range(0, self.pairs.shape[0]):
#         if i % inc == 0: pair_mask[i] = i
#     pruned = self.pairs[pair_mask, :]
#
#     plt.plot([pruned[:, PAIR_TIME_IDX], pruned[:, PAIR_TIME_IDX + 1]],
#              [pruned[:, FREQ_IDX], pruned[:, FREQ_IDX + 1]], 'w-')
#     plt.show(block=False)


# Get 2d peaks from a spectrogram
# @staticmethod
def get_2d_peaks(sxx, amp_min=DEFAULT_AMP_MIN):
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our filter shape
    local_max = maximum_filter(sxx, footprint=neighborhood) == sxx
    background = (sxx == 0)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    # Boolean mask of arr2D with True at peaks
    detected_peaks = local_max ^ eroded_background

    # extract peaks
    amps = sxx[detected_peaks]
    j, i = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = [x for x in peaks if x[2] > amp_min]  # freq, time, amp
    # if len(peaks_filtered) <= 1:
    #     print("Only {} peaks found for {} amp min!".format(len(peaks_filtered), amp_min))

    # get indices for frequency and time
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]

    return frequency_idx, time_idx


# Apply the Short-Time Fourier Transform
def transform_stft(samples, fs=DEFAULT_FS,
                   wsize=DEFAULT_WINDOW_SIZE,
                   wratio=DEFAULT_OVERLAP_RATIO):
    freq, times, spect = signal.stft(
        samples,
        nfft=wsize,
        fs=fs,
        window='hann',
        nperseg=int(wsize),
        noverlap=int(wsize * wratio))
    return freq, times, spect


def plot_stft(freq, time, spectrogram, name='STFT Magnitude'):
    plt.figure()
    plt.pcolormesh(time, freq, np.abs(spectrogram), vmin=0, vmax=5)
    plt.title(name)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)


# Search a single column for data within the tolerance
def search_col(data, query, tol=0):
    if tol == 0:
        match_idx = np.asarray(np.where(data == query), int)[0]
    else:
        low = np.where(data > query - tol)[0]
        high = np.where(data < query + tol)[0]
        match_idx = np.asarray(np.intersect1d(low, high), int)
    return match_idx


# Read audio as a time-signal. Mix stereo channels evenly if necessary
def read_audio(filename):
    fs, sound = read(filename, mmap=False)
    if len(sound.shape) > 1 and sound.shape[1] > 1:
        num_channels = sound.shape[1]
        mono_mix = np.zeros((num_channels, 1))
        mono_mix.fill(1 / num_channels)
        sound = sound @ mono_mix
        sound = sound[:, 0]
    return sound, fs


def main():
    audio_lb, sr = fileutils.load_audio('t1', './bin/', sr=DEFAULT_FS)
    audio, fs = read_audio('./bin/t1.wav')
    print(audio.dtype)
    zxx_lb = librosa.stft(audio_lb,
                          n_fft=DEFAULT_WINDOW_SIZE,
                          win_length=DEFAULT_WINDOW_SIZE,
                          hop_length=int(DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO),
                          window='hann'
                          )
    freqs, times, zxx_sci = transform_stft(audio, fs=fs, wratio=DEFAULT_OVERLAP_RATIO, wsize=DEFAULT_WINDOW_SIZE)
    plot_stft(freqs, times, zxx_sci)
    D = librosa.amplitude_to_db(np.abs(zxx_lb), ref=np.max)
    plt.figure()
    librosa.display.specshow(D, x_axis='time', y_axis='linear',
                             sr=sr, hop_length=(DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO))
    plt.show()
    return None


if __name__ == '__main__':
    main()
