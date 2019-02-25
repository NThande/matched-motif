import numpy as np
from librosa import stft
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure,
                                      iterate_structure, binary_erosion)

import fileutils
import visutils as vis
import config as cfg

# Modified from Dejavu Fingerprinting System (as of 11/21/18)
FREQ_IDX = 0
PEAK_TIME_IDX = 1
PAIR_TIME_IDX = 2
PAIR_TDELTA_IDX = 4
DEFAULT_FAN_VALUE = 15
DEFAULT_AMP_MIN = 10
PEAK_NEIGHBORHOOD_SIZE = 20
MIN_TIME_DELTA = 0
MAX_TIME_DELTA = 200


# FFT the channel, log transform output, find local maxima, then return
# locally sensitive hashes.
def fingerprint(audio,
                wsize=cfg.WINDOW_SIZE,
                wratio=cfg.OVERLAP_RATIO,
                fan_value=DEFAULT_FAN_VALUE,
                amp_min=DEFAULT_AMP_MIN):
    # FFT the signal and extract frequency components
    sxx = stft(audio,
               n_fft=wsize,
               win_length=wsize,
               hop_length=int(wsize * wratio),
               window='hann'
               )

    sxx = np.abs(sxx)

    # find local maxima
    peaks_f, peaks_t = get_2d_peaks(sxx, amp_min=amp_min)
    peaks = np.stack((peaks_f, peaks_t), axis=1)
    pairs = generate_pairs(peaks, fan_value)
    return peaks, pairs


# Generate peak-pairs based on locally-sensitive target zone
def generate_pairs(peaks, fan_value=DEFAULT_FAN_VALUE):
    # peaks = np.unique(peaks, axis=0)
    num_peaks = peaks.shape[0]
    pairs = np.zeros((0, 5))

    for i in range(num_peaks):
        for j in range(1, fan_value):
            if (i + j) < num_peaks:

                freq1 = peaks[i, FREQ_IDX]
                freq2 = peaks[i + j, FREQ_IDX]
                t1 = peaks[i, PEAK_TIME_IDX]
                t2 = peaks[i + j, PEAK_TIME_IDX]
                t_delta = t2 - t1

                if MIN_TIME_DELTA <= t_delta <= MAX_TIME_DELTA:
                    pairs = np.vstack((pairs, np.array([freq1, freq2, t1, t2, t_delta])))

    # Return dummy entry
    if pairs.shape[0] == 0:
        print("No pairs found, only {} peaks".format(peaks.shape[0]))
        return np.zeros((1, 5))
    pairs = np.unique(pairs, axis=0)
    return pairs


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

    # get indices for frequency and time
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]

    return frequency_idx, time_idx


# Search fingerprint song_fp for entries from fingerprint sample_fp
def linear_search(db_pairs, query_pairs, **kwargs):
    matches = np.zeros(query_pairs.shape[0])
    for i in range(0, query_pairs.shape[0]):
        _, matches[i] = linear_query(db_pairs, query_pairs[i], **kwargs)
    return matches


# Query the pair table for one pair
def linear_query(db_pairs, query, delta_tol=10, f_tol=50):
    t_delta_matches = search_col(db_pairs[:, PAIR_TDELTA_IDX], query[PAIR_TDELTA_IDX], delta_tol)
    t_pairs = db_pairs[t_delta_matches]

    f1_matches = search_col(t_pairs[:, FREQ_IDX], query[FREQ_IDX], f_tol)
    f1_pairs = t_pairs[f1_matches]

    f2_matches = search_col(f1_pairs[:, FREQ_IDX + 1], query[FREQ_IDX + 1], f_tol)
    f2_pairs = f1_pairs[f2_matches]

    num_matches = f2_pairs.shape[0]
    tf2_idx = (t_delta_matches[f1_matches])[f2_matches]
    return tf2_idx, num_matches


# Search a single column for data within the tolerance
def search_col(data, query, tol=0):
    if tol == 0:
        match_idx = np.asarray(np.where(data == query), int)[0]
    else:
        low = np.where(data > query - tol)[0]
        high = np.where(data < query + tol)[0]
        match_idx = np.asarray(np.intersect1d(low, high), int)
    return match_idx


def main():
    audio, fs = fileutils.load_audio('t1', './bin/')
    peaks, pairs = fingerprint(audio)
    sxx = stft(audio,
               n_fft=cfg.WINDOW_SIZE,
               win_length=cfg.WINDOW_SIZE,
               hop_length=int(cfg.WINDOW_SIZE * cfg.OVERLAP_RATIO),
               window='hann')
    vis.plot_stft(sxx, fs=fs, frames=False)
    vis.plot_peaks(peaks)
    vis.plot_pairs(peaks, pairs)
    vis.plot_stft_with_pairs(sxx, peaks, pairs)
    vis.show()
    return None


if __name__ == '__main__':
    main()
