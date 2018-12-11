import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io.wavfile import read
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure,
                                      iterate_structure, binary_erosion)

# Modified from Dejavu Fingerprinting System (as of 11.21.18)
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


class FingerPrint:
    def __init__(self, sound, Fs=DEFAULT_FS):
        self.sound = sound
        self.peaks = None
        self.pairs = None
        self.fs = Fs
        self.hasFingerPrint = False

    # FFT the channel, log transform output, find local maxima, then return
    # locally sensitive hashes.
    def generate_fingerprint(self, Fs=DEFAULT_FS,
                             wsize=DEFAULT_WINDOW_SIZE,
                             wratio=DEFAULT_OVERLAP_RATIO,
                             fan_value=DEFAULT_FAN_VALUE,
                             amp_min=DEFAULT_AMP_MIN,
                             to_plot=False):
        self.hasFingerPrint = False
        # FFT the signal and extract frequency components
        frequencies, times, spectrogram = transform_stft(self.sound, Fs, wsize, wratio)
        if to_plot:
            plt.figure()
            plt.pcolormesh(times, frequencies, np.abs(spectrogram), vmin=0, vmax=5)
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show(block=False)

        spectrogram[spectrogram == -np.inf] = 0
        spectrogram = np.abs(spectrogram)

        # find local maxima
        peaks_f, peaks_t = self.get_2d_peaks(spectrogram, amp_min=amp_min)
        peaks = np.zeros([len(peaks_f), 2])
        peaks[:, 0] = frequencies[peaks_f]
        peaks[:, 1] = times[peaks_t]
        self.peaks = peaks
        if to_plot:
            self.plot_peaks()
        pairs = self.generate_pairs(fan_value)
        self.pairs = pairs
        if to_plot:
            self.plot_pairs()
        self.hasFingerPrint = True

        return self.peaks, self.pairs

    # Generate peak-pairs based on locally-sensitive target zone
    def generate_pairs(self, fan_value=DEFAULT_FAN_VALUE, tol=0):
        peaks = np.unique(self.peaks, axis=0)
        # if peaks.shape[0] <= 1:
        #     print("Too few peaks to pair: {}".format(peaks.shape[0]))
        #     return np.asarray([peaks[0, FREQ_IDX],
        #                        peaks[0, FREQ_IDX],
        #                        peaks[0, PEAK_TIME_IDX],
        #                        peaks[0, PEAK_TIME_IDX], 0])
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

        if pairs.shape[0] == 0:
            print("No pairs found, only {} peaks".format(peaks.shape[0] - 1))
            self.generate_pairs(fan_value, tol + 1)
            return pairs
        pairs = np.unique(pairs, axis=0)
        return pairs

    # Query the pair table
    def search_for_pair(self, query):
        t_delta_tol = 10
        t_delta_matches = search_col(self.pairs[:, PAIR_TDELTA_IDX], query[PAIR_TDELTA_IDX], t_delta_tol)
        t_pairs = self.pairs[t_delta_matches]

        f_tol = 50
        f1_matches = search_col(t_pairs[:, FREQ_IDX], query[FREQ_IDX], f_tol)
        f1_pairs = t_pairs[f1_matches]

        f2_matches = search_col(f1_pairs[:, FREQ_IDX + 1], query[FREQ_IDX + 1], f_tol)
        f2_pairs = f1_pairs[f2_matches]

        num_matches = f2_pairs.shape[0]
        tf2_idx = (t_delta_matches[f1_matches])[f2_matches]
        return tf2_idx, num_matches

    # Plot Spectrogram peaks
    def plot_peaks(self):
        if self.peaks is None:
            return
        plt.figure()
        plt.plot(self.peaks[:, PEAK_TIME_IDX], self.peaks[:, FREQ_IDX], 'rx')
        plt.title('STFT Peaks')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show(block=False)

    # Plot spectogram peaks and one pair from every inc pairs
    def plot_pairs(self, inc=50):
        if self.pairs is None:
            return
        pair_mask = np.zeros(self.pairs.shape[0]).astype(int)
        for i in range(0, self.pairs.shape[0]):
            if i % inc == 0: pair_mask[i] = i
        pruned = self.pairs[pair_mask, :]

        plt.figure()
        plt.plot(self.peaks[:, PEAK_TIME_IDX], self.peaks[:, FREQ_IDX], 'rx')
        plt.plot([pruned[:, PAIR_TIME_IDX], pruned[:, PAIR_TIME_IDX + 1]],
                 [pruned[:, FREQ_IDX], pruned[:, FREQ_IDX + 1]], 'b-')

        plt.plot(pruned[:, PAIR_TIME_IDX], pruned[:, FREQ_IDX], 'kx')
        plt.plot(pruned[:, PAIR_TIME_IDX + 1], pruned[:, FREQ_IDX + 1], 'k*')
        plt.title('Peak Pairs')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show(block=False)

    # Get 2d peaks from a spectrogram
    @staticmethod
    def get_2d_peaks(spect, amp_min=DEFAULT_AMP_MIN):
        struct = generate_binary_structure(2, 1)
        neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

        # find local maxima using our filter shape
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
        if len(peaks_filtered) <= 1:
            print("Only {} peaks found for {} amp min!".format(len(peaks_filtered), amp_min))

        # get indices for frequency and time
        frequency_idx = [x[1] for x in peaks_filtered]
        time_idx = [x[0] for x in peaks_filtered]

        return frequency_idx, time_idx


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


# Apply the Short-Time Fourier Transform
def transform_stft(samples, fs=DEFAULT_FS,
                   wsize=DEFAULT_WINDOW_SIZE,
                   wratio=DEFAULT_OVERLAP_RATIO):
    freqs, times, spect = signal.stft(
        samples,
        nfft=wsize,
        fs=fs,
        window='hann',
        nperseg=int(wsize),
        noverlap=int(wsize * wratio))
    return freqs, times, spect


# Search a single column for data within the tolerance
def search_col(data, query, tol=0):
    if tol == 0:
        match_idx = np.asarray(np.where(data == query), int)[0]
    else:
        low = np.where(data > query - tol)[0]
        high = np.where(data < query + tol)[0]
        match_idx = np.asarray(np.intersect1d(low, high), int)
    return match_idx
