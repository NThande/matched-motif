import FingerPrint as fp
import matplotlib as plt
import numpy as np

FREQ_IDX = 0
PEAK_TIME_IDX = 1
PAIR_TIME_IDX = 2
PAIR_TDELTA_IDX = 4

# Test basic Shazam-style identification
sound, r = fp.read_audio('./main/bin/unique/hello_train.wav')
sound_peaks, sound_data = fp.fingerprint(sound, Fs=r)


# Basic test
audio, r = fp.read_audio('./main/bin/unique/hello_train.wav')
audio_fp = fp.FingerPrint(audio, r)
audio_fp.generate_fingerprint()
audio_fp.plot_peaks()
audio_fp.plot_pairs()
_, num_self_matches = audio_fp.search_for_pair(audio_fp.pairs[1, :])
print("Matches for Pair 0: ", num_self_matches)
plt.show()



sample_length = 2
for i in range(3, 4):
    sound, r = fp.read_audio('./main/bin/t{}.wav'.format(i + 1))
    sound_data = fp.fingerprint(sound, r)
    sample = sound[0:(r * sample_length)]
    sample_data = fp.fingerprint(sample, r)
    print(sample_data.shape)
    print(sound_data.shape)
    print(sample_data[0])
    sample_matches = np.zeros([sample_data.shape[0], 2])
    sample_matches[:, 0] = np.arange(0, sample_data.shape[0])
    for sample_pair in sample_data:
        matches = fp.search_for_pair(sound_data, sample_pair)
        print(matches.shape[1])
plt.show()
