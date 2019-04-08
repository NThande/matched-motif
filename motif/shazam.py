import hashlib
import numpy as np
from librosa import stft
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure,
                                      iterate_structure, binary_erosion)
import fileutils
import visutils as vis
import config as cfg

# Modified from Dejavu Fingerprinting System (as of 11/21/18)
FREQ_IDX = cfg.FREQ_IDX
PEAK_TIME_IDX = FREQ_IDX + 1
PAIR_TIME_IDX = cfg.PAIR_TIME_IDX
PAIR_TDELTA_IDX = cfg.PAIR_TIME_IDX + 2
DEFAULT_FAN_VALUE = 15
DEFAULT_AMP_MIN = 10
PEAK_NEIGHBORHOOD_SIZE = 20
MIN_TIME_DELTA = 0
MAX_TIME_DELTA = 200
FINGERPRINT_REDUCTION = 10


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
    # sxx = 10 * np.log10(sxx)
    sxx[sxx == -np.inf] = 0  # replace infs with zeros

    # find local maxima
    peaks_f, peaks_t = get_2d_peaks(sxx, amp_min=amp_min)
    peaks = np.stack((peaks_f, peaks_t), axis=1)
    pairs_hash, pairs_matrix = generate_pairs(peaks, fan_value)
    return pairs_hash, pairs_matrix, peaks


# Generate peak-pairs based on locally-sensitive target zone
def generate_pairs(peaks, fan_value=DEFAULT_FAN_VALUE):
    # peaks = np.unique(peaks, axis=0)
    num_peaks = peaks.shape[0]
    pairs_matrix = np.zeros((0, 5))
    pairs_hash = {}
    encoding = 'utf-8'

    for i in range(num_peaks):
        for j in range(1, fan_value):
            if (i + j) < num_peaks:

                freq1 = peaks[i, FREQ_IDX]
                freq2 = peaks[i + j, FREQ_IDX]
                f_delta = freq2 - freq1
                t1 = peaks[i, PEAK_TIME_IDX]
                t2 = peaks[i + j, PEAK_TIME_IDX]
                t_delta = t2 - t1

                if MIN_TIME_DELTA <= t_delta <= MAX_TIME_DELTA:
                    pairs_matrix = np.vstack((pairs_matrix, np.array([freq1, freq2, t1, t2, t_delta])))
                    h = hashlib.sha1(
                        "{}|{}|{}".format(
                            str(freq1).encode(encoding),
                            str(t_delta).encode(encoding),
                            str(f_delta).encode(encoding)
                        ).encode(encoding))
                    this_hash = h.hexdigest()[0:FINGERPRINT_REDUCTION]
                    pairs_hash[this_hash] = t1

    # Return dummy entry
    if pairs_matrix.shape[0] == 0:
        print("No pairs found, only {} peaks".format(peaks.shape[0]))
        return {}, np.zeros((1, 5))
    pairs_matrix = np.unique(pairs_matrix, axis=0)
    return pairs_hash, pairs_matrix


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


def hash_search(db_hashes, query_hashes):
    offsets_list = []
    for query in query_hashes:
        if query in db_hashes:
            t_offset = db_hashes[query] - query_hashes[query]
            offsets_list.append(t_offset)
    offsets = np.array(offsets_list)
    buckets, _ = np.histogram(offsets)
    matches = np.max(buckets)
    return matches


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
    name = 'genre_test_3'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    pairs_hash, pairs, peaks  = fingerprint(audio)

    sxx = stft(audio,
               n_fft=cfg.WINDOW_SIZE,
               win_length=cfg.WINDOW_SIZE,
               hop_length=int(cfg.WINDOW_SIZE * cfg.OVERLAP_RATIO),
               window='hann')
    sxx = np.abs(sxx)
    sxx = 10 * np.log10(sxx)
    sxx[sxx == -np.inf] = 0  # replace infs with zeros

    seg_hash_one, _, _ = fingerprint(audio[0 * fs: 3 * fs])
    seg_hash_two, _, _ = fingerprint(audio[0 * fs: 3 * fs])
    print(hash_search(seg_hash_one, seg_hash_two))

    vis.plot_stft(sxx, fs=fs, frames=False)
    vis.plot_peaks(peaks)
    vis.plot_pairs(peaks, pairs)
    vis.plot_stft_with_pairs(sxx, peaks, pairs)
    vis.show()
    return None


if __name__ == '__main__':
    main()
