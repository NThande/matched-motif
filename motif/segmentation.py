import fileutils
import numpy as np
import librosa as lb
import config as cfg


# Choose a segmentation method from a given input
def segment(audio, fs, length=2, overlap=cfg.OVERLAP_RATIO, method='regular', **kwargs):
    segments = None
    if method is 'regular':
        segments = segment_regular(audio, fs, length=length, overlap=overlap, **kwargs)
    elif method is 'onset' or method is 'beat':
        segments = segment_onset(audio, fs, length=length, overlap=overlap, method=method, **kwargs)
    else:
        print("Unrecognized segmentation method: {}".format(method))
    return segments


# Create regular segments of fixed length with an overlap ratio in seconds
def segment_regular(audio, fs, length, overlap, **kwargs):
    total_length = audio.shape[0] / fs
    segment_list = []
    k = 0
    while k < (total_length - length):
        segment_list.append(k)
        k += length * (1 - overlap)
    segments = np.asarray(segment_list)
    return segments


# Detect onsets and merge segments forwards to meet minimum window length
def segment_onset(audio, fs,
                  length,
                  overlap,
                  method='onset',
                  prune=False,
                  fill_space=False):
    if method is 'onset':
        onsets = lb.onset.onset_detect(audio, fs, hop_length=cfg.WINDOW_SIZE, units='time', backtrack=True)
        # onsets = lb.onset.onset_detect(audio, fs, hop_length=cfg.WINDOW_SIZE, units='time', backtrack=False)
    elif method is 'beat':
        _, onsets = lb.beat.beat_track(audio, fs, hop_length=cfg.WINDOW_SIZE, units='time')
    else:
        return None

    # # If no onsets are found, default to regular segmentation
    # if onsets.shape[0] == 0:
    #     return segment_regular(audio, fs, length, overlap)

    if onsets[0] != 0.:
        onsets = np.insert(onsets, 0, 0)

    num_onsets = onsets.shape[0]

    # Prune onsets that are too close to the previous onset or the end of the audio
    mask = np.zeros(onsets.shape)
    if prune is False:
        mask = np.ones(onsets.shape)
    min_diff = length * (1 - overlap)
    audio_end = audio.shape[0] / fs
    prev_onset = onsets[0]
    for i in range(1, num_onsets):
        if onsets[i] + length > audio_end:
            mask[i:num_onsets] = 0
            break
        if onsets[i] - min_diff > prev_onset:
            mask[i] = 1
            prev_onset = onsets[i]

    onsets = onsets[mask == 1]

    # Add regularly spaced windows in periods without onsets
    if fill_space is True:
        num_onsets = onsets.shape[0]
        add_idx = []
        add_seg = []
        for i in range(1, num_onsets):
            if onsets[i] - length > onsets[i - 1]:
                add_onset = onsets[i - 1] + min_diff
                while add_onset < onsets[i] - min_diff:
                    add_idx.append(i)
                    add_seg.append(add_onset)
                    add_onset = add_onset + min_diff

        onsets = np.insert(onsets, add_idx, add_seg)

    return onsets


def main():
    name = 't1'
    directory = "./bin/"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    print(segment_onset(audio, fs))
    print(segment_onset(audio, fs, method='beat'))


if __name__ == '__main__':
    main()
