import numpy as np
import FingerPrint as fp
import matplotlib.pyplot as plt


# Basic test
def basic_fingerprint_test(filename):
    audio, r = fp.read_audio(filename)
    audio_fp = fp.FingerPrint(audio, r)
    audio_fp.generate_fingerprint(to_plot=True, plot_inc=500)
    _, num_self_matches = audio_fp.search_for_pair(audio_fp.pairs[1, :])
    return num_self_matches


def test_find_self(sample_length, t_start, t_end):
    self_match_ratio = np.zeros(t_end - t_start)
    count = 0
    for i in range(t_start, t_end):
        sound, r = fp.read_audio('./main/bin/t{}.wav'.format(i + 1))
        sound_fp = fp.FingerPrint(sound, r)
        sample = sound[0:(r * sample_length)]
        sample_fp = fp.FingerPrint(sample, r)

        for sample_pair in sample_fp.pairs:
            _, matches = sound_fp.search_for_pair(sample_pair)
            self_match_ratio[count] += matches
        self_match_ratio[count] = self_match_ratio[count] / sample_fp.pairs.shape[0]
        count += 1
    return self_match_ratio

path = './main/bin/unique/'
file_type = '.wav'
audio_file = 'hello_train'
basic_fingerprint_test(path + audio_file + file_type)
plt.show()
