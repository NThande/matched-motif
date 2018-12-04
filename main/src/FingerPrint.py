import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io.wavfile import (read, write)
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
PEAK_SORT = True


def transform_stft(samples, Fs=DEFAULT_FS,
                wsize=DEFAULT_WINDOW_SIZE,
                wratio=DEFAULT_OVERLAP_RATIO):
    freqs, times, spect = signal.stft(
        samples,
        nfft=wsize,
        fs=Fs,
        window='hann',
        nperseg=int(wsize),
        noverlap=int(wsize * wratio))
    return freqs, times, spect


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

    # apply log transform since specgram() returns linear array
    spect[spect == -np.inf] = 0
    spect = np.abs(spect)
    # print(spect.shape)

    # find local maxima
    peaks_f, peaks_t = get_2d_peaks(spect, amp_min=amp_min)
    peaks = np.zeros([len(peaks_f), 2])
    peaks[:, 0] = freqs[peaks_f]
    peaks[:, 1] = times[peaks_t]

    pairs = generate_pairs(peaks, fan_value)
    # plot_peaks(peaks)
    # plot_pairs(peaks, pairs, 50)

    return peaks, pairs


def plot_peaks(peaks):
    plt.figure()
    plt.plot(peaks[:, PEAK_TIME_IDX], peaks[:, FREQ_IDX], 'rx')
    plt.title('STFT Peaks')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)


def plot_pairs(peaks, pairs, inc):
    pair_mask = np.zeros(pairs.shape[0]).astype(int)
    for i in range(0, pairs.shape[0]):
        if i % inc == 0: pair_mask[i] = i
    # print(pair_mask)
    pruned = pairs[pair_mask, :]
    # print(pruned.shape)
    plt.figure()
    plt.plot(peaks[:, PEAK_TIME_IDX], peaks[:, FREQ_IDX], 'rx')
    plt.plot([pruned[:, PAIR_TIME_IDX], pruned[:, PAIR_TIME_IDX + 1]],
             [pruned[:, FREQ_IDX], pruned[:, FREQ_IDX + 1]], 'b-')

    plt.plot(pruned[:, PAIR_TIME_IDX], pruned[:, FREQ_IDX], 'kx')
    plt.plot(pruned[:, PAIR_TIME_IDX + 1], pruned[:, FREQ_IDX + 1], 'k*')
    plt.title('Peak Pairs')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)


def get_2d_peaks(spect, amp_min=DEFAULT_AMP_MIN):
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

    return frequency_idx, time_idx


def generate_pairs(peaks, fan_value=DEFAULT_FAN_VALUE):

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

                if MIN_TIME_DELTA <= t_delta <= MAX_TIME_DELTA:
                    pairs = np.vstack((pairs, np.array([freq1, freq2, t1, t2, t_delta])))

    # print(pairs.shape)
    pairs = np.unique(pairs, axis=0)
    return pairs


# Query the pair table
def search_for_pair(pairs, query):
    t_delta_eq = np.where(pairs[:, PAIR_TDELTA_IDX] == query[PAIR_TDELTA_IDX])[0]
    t_delta_matches = t_delta_eq
    # t_delta_tol = 0.5
    # t_delta_low = np.where(pairs[:, PAIR_TDELTA_IDX] > query[PAIR_TDELTA_IDX] - t_delta_tol)[0]
    # t_delta_high = np.where(pairs[:, PAIR_TDELTA_IDX] < query[PAIR_TDELTA_IDX] + t_delta_tol)[0]
    # t_delta_matches = np.intersect1d(t_delta_low, t_delta_high)
    # t_delta_matches = np.asarray(t_delta_matches, int)

    t_pairs = pairs[t_delta_matches]
    # print(t_pairs.shape)

    f1_tol = 50
    f1_low = np.where(t_pairs[:, FREQ_IDX] > query[FREQ_IDX] - f1_tol)
    f1_high = np.where(t_pairs[:, FREQ_IDX] < query[FREQ_IDX] + f1_tol)
    f1_matches = np.intersect1d(f1_low, f1_high)
    f1_pairs = t_pairs[f1_matches]
    f1_matches = np.asarray(f1_matches, int)
    # print(f1_pairs.shape)

    f2_tol = 50
    f2_low = np.where(f1_pairs[:, FREQ_IDX + 1] > query[FREQ_IDX + 1] - f2_tol)
    f2_high = np.where(f1_pairs[:, FREQ_IDX + 1] < query[FREQ_IDX + 1] + f2_tol)
    f2_matches = np.intersect1d(f2_low, f2_high)
    f2_matches = np.asarray(f2_matches, int)
    tf2_pairs = f1_pairs[f2_matches]
    tf2_idx = (t_delta_matches[f1_matches])[f2_matches]
    # print(tf2_pairs.shape)

    pass
    return tf2_idx, tf2_pairs.shape[0]


# Read audio as a time-signal. Mix stereo channels evenly if necessary
def read_audio(filename):
    fs, sound = read(filename, mmap=False)
    # print(sound.shape)
    if len(sound.shape) > 1 and sound.shape[1] > 1:
        num_channels = sound.shape[1]
        mono_mix = np.zeros((num_channels, 1))
        mono_mix.fill(1 / num_channels)
        sound = sound @ mono_mix
        sound = sound[:, 0]
    # print(sound.shape)
    return sound, fs


# Test basic Shazam-style identification
sound, r = read_audio('./main/bin/unique/hello_train.wav')
# sound, r = read_audio('./main/bin/t1.wav')

# print(len(sound.shape))
# sound = sound[:, 0]
# print(sound.shape)
# Rough monophonic mix

sound_peaks, sound_data = fingerprint(sound, Fs=r)
# sound_peaks, sound_data = fingerprint(sound, Fs=r)
sample, r = read_audio('./main/bin/unique/hello_test.wav')
# sample = sample[:, 0]
sample_peaks, sample_data = fingerprint(sample, Fs=r)
# sample_peaks, sample_data = fingerprint(sample, Fs=r)

plot_peaks(sound_peaks)
plot_pairs(sound_peaks, sound_data, 100)
# plot_peaks(sample_peaks)
# plot_pairs(sample_peaks, sample_data, 40)
#print(sound_data.shape)
# print(sample_data.shape)

print(sample_data[0].astype(int))
match_vect = np.zeros(sample_data.shape[0])
for i in range(0, sample_data.shape[0]):
    _, match_vect[i] = search_for_pair(sound_data, sample_data[i])
matches = np.average(match_vect)

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

# How does the length of the audio affect our number of matches?
snip_end = 13
snip_matches = np.zeros(6)
snip_lengths = np.arange(13, 2, -2)
# print(snip_lengths)
# snip_lengths = np.insert(snip_lengths, 0, sound.shape[0] / r, axis=0)
# print(snip_lengths)
# snip_matches[0] = matches
samp_hits = np.zeros(sound_data.shape[0])
for i in range(0, snip_matches.shape[0] - 1):
    snippet_peaks, snippet_data = fingerprint(sound[i*r: snip_end*r], Fs=r)
    match_vect = np.zeros(sample_data.shape[0])
    for j in range(0, sample_data.shape[0]):
        match_idx, match_vect[j] = search_for_pair(snippet_data, sample_data[j])
        samp_hits[match_idx] += 1
    snip_end = snip_end - 1
    snip_matches[i] = np.average(match_vect)
plt.figure()
print(snip_lengths)
print(snip_matches)
plt.plot(snip_lengths, snip_matches, 'rx-')
plt.xlabel("Snippet Length")
plt.ylabel("Number of matches in database")
plt.title("Number of matches for fixed snippet vs length of overall fingerprint track")

plt.figure()
plt.plot(sound_data[:, PAIR_TIME_IDX], samp_hits, 'rx', label='Pair End Time')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Number of matches")
plt.title("Matches per pair using Fixed Sample")

# plt.show()

# Apply a sliding window on the sound itself
sound_length = np.ceil(sound.shape[0] / r)
snap_length = 2
snap_num = int(sound_length - snap_length)
snap_windows = np.arange(0, snap_num)
snap_matches = np.zeros(snap_num)
snap_hits = np.zeros(sound_data.shape[0])
for i in range(0, snap_num):
    snap_start = i * r
    snap_end = (i + snap_length) * r
    if snap_end > sound.shape[0] :
        snap_end = sound.shape[0]
    snap = sound[snap_start : snap_end]
    snap_peaks, snap_data = fingerprint(snap, Fs=r)
    match_vect = np.zeros(snap_data.shape[0])
    for j in range(0, snap_data.shape[0]):
        match_idx, match_vect[j] = search_for_pair(sound_data, snap_data[j])
        snap_hits[match_idx] += 1
    snap_matches[i] = np.average(match_vect)

print(snap_hits)
plt.figure()
print(snap_matches)
plt.plot(snap_windows, snap_matches, 'rx-')
plt.xlabel("Snapshot Starting Point (s)")
plt.ylabel("Number of matches in database")
plt.title("Average number of matches for Fixed Length Sliding Windows")

max_samp_idx = np.argmax(snap_matches)
max_samp_num = snap_windows[max_samp_idx]
max_sample = sound[max_samp_num * r + 1: (max_samp_num + snap_length) * r]
write('./main/bin/unique/test.wav', r, max_sample)

plt.figure()
plt.plot(sound_data[:, PAIR_TIME_IDX], snap_hits, 'rx', label='Pair Start Time')
#plt.plot(sound_data[:, PAIR_TIME_IDX + 1], snap_hits, 'bx', label='Pair Start Time')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Number of matches")
plt.title("Matches per pair using Fixed Length Sliding Window")
plt.show()

