import numpy as np
import fileutils
import fingerprint as fp


# Search fingerprint song_fp for entries from fingerprint sample_fp
def basic_search(song_fp, sample_fp):
    matches = np.zeros(sample_fp.pairs.shape[0])
    for i in range(0, sample_fp.pairs.shape[0]):
        _, matches[i] = song_fp.search_for_pair(sample_fp.pairs[i])
    return matches


# Search for a sample using different windows of the song centered on a point
def shrinking_search(song_fp, sample_fp, start_time, center_time):
    sound = song_fp.sound
    r = song_fp.fs

    snip_start = start_time
    if snip_start < 0:
        snip_start = 0
    snip_end = 2 * center_time - start_time
    if snip_end > sound.shape[0] / r:
        snip_end = sound.shape[0] / r

    samp_hits = np.zeros(song_fp.pairs.shape[0])
    num_iter = int(np.floor((snip_end - snip_start) / 2))
    snip_lengths = np.zeros(num_iter)
    snip_matches = np.zeros(num_iter)

    for i in range(0, num_iter):
        snippet_fp = fp.FingerPrint(sound[snip_start * r: snip_end * r], Fs=r)
        snippet_fp.generate_fingerprint()
        match_vect = np.zeros(sample_fp.pairs.shape[0])
        for j in range(0, sample_fp.pairs.shape[0]):
            match_idx, match_vect[j] = snippet_fp.search_for_pair(sample_fp.pairs[j])
            samp_hits[match_idx] += 1
        if (snip_start < center_time):
            snip_start += 1
        if (snip_end > center_time):
            snip_end -= 1
        snip_matches[i] = np.average(match_vect)
        snip_lengths[i] = snip_end - snip_start
    #
    # if to_plot:
    #     plt.figure()
    #     plt.plot(snip_lengths, snip_matches, 'rx-')
    #     plt.xlabel("Snippet Length")
    #     plt.ylabel("Number of matches in database")
    #     plt.title("Number of matches for fixed snippet vs length of overall fingerprint track")

    return samp_hits, snip_matches, snip_lengths


# Apply a sliding window of a part of the song to the rest of the song
def thumbnail(song_fp, window_length=2):
    sound = song_fp.sound
    r = song_fp.fs
    seg_coeff = 1
    song_length = np.ceil(sound.shape[0] / r)
    snap_num = int((song_length - window_length) * 1 / seg_coeff)
    snap_windows = np.arange(0, snap_num) * seg_coeff
    snap_matches = np.zeros(snap_num)
    snap_hits = np.zeros(song_fp.pairs.shape[0])

    for i in range(0, snap_num):
        snap_start = int(i * r * seg_coeff)
        snap_end = int(snap_start + (window_length * r))

        if snap_end > sound.shape[0]:
            snap_end = sound.shape[0]
        snap = sound[snap_start: snap_end]
        snap_fp = fp.FingerPrint(snap, r)
        snap_peaks, snap_pairs = snap_fp.generate_fingerprint()
        match_vect = np.zeros(snap_pairs.shape[0])

        for j in range(0, snap_pairs.shape[0]):
            match_idx, match_vect[j] = song_fp.search_for_pair(snap_pairs[j])
            snap_hits[match_idx] += 1
        snap_matches[i] = np.average(match_vect)
        print("Completed Window {} / {}".format(i, snap_num))
    snap_matches = snap_matches / np.max(snap_matches)
    max_samp_idx = np.argmax(snap_matches)
    max_samp_num = snap_windows[max_samp_idx]
    max_sample = sound[max_samp_num * r + 1: (max_samp_num * r) + int((window_length) * r)]
    #
    # if to_plot:
    #     fig = plt.gcf()
    #     fig.set_size_inches(13.5, 10.5, forward=True)
    #     plt.rcParams["figure.figsize"] = [16, 4]
    #     plt.grid()
    #     plt.figure()
    #     plt.plot(snap_windows, snap_matches, 'rx-')
    #     plt.xlabel("Snippet Starting Point (s)")
    #     plt.ylabel("Similarity Score")
    #     plt.title("Self-Similarity Scores Using New Algorithm".format(window_length))
    #
    #     if labels is not None:
    #         axes = plt.gca()
    #         y_max = axes.get_ylim()[1]
    #         axes.set_xlim(axes.get_xlim()[0] - 1, axes.get_xlim()[1] + 1)
    #         for i in range(0, labels.shape[0] - 1):
    #             plt.axvspan(labels.Time[i], labels.Time[i + 1], alpha=0.2, color=labels.Color[i],
    #                         linestyle='-.', label='Motif {}'.format(labels.Event[i]))
    #             plt.grid()
    #
    # if write_name is not None:
    #     write(write_name, r, max_sample)
    #
    # if get_info:
    #     return max_sample, snap_windows, snap_matches, snap_hits
    print("Auto Window complete")
    return max_sample, snap_matches, snap_windows


def main():
    name = 't3_train'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    audio_labels = fileutils.load_labels(name, label_dir=directory)

    audio_fp = fp.FingerPrint(audio, fs)
    audio_fp.generate_fingerprint()

    _, windows, matches = thumbnail(song_fp=audio_fp,
                                    window_length=2)

    # plt.rc('font', size=15)  # controls default text sizes
    # fig = plt.gcf()
    # fig.set_size_inches(13.5, 10.5, forward=True)
    # plt.rcParams["figure.figsize"] = [16, 4]
    # plt.grid()

    sliding_lengths = np.arange(1, 3.01, 1.0)
    windows_coll = []
    matches_coll = []

    for i in range(0, sliding_lengths.shape[0]):
        window_len = sliding_lengths[i]
        _, windows, matches = thumbnail(song_fp=audio_fp,
                                        window_length=window_len)
        windows_coll.append(windows)
        matches_coll.append(matches)

    # plt.figure()
    # plt.title("Similarity Scores for Various Window Lengths")
    # plt.xlabel("Snippet Starting Point (s)")
    # plt.ylabel("Similarity Score")
    # count = 0
    # for i in range(0, len(windows_coll)):
    #     plt.plot(windows_coll[i], matches_coll[i], 'C{}-'.format(i), label="{} s".format(str(sliding_lengths[i])))
    #
    # if audio_labels is not None:
    #     labels = audio_labels
    #     axes = plt.gca()
    #     y_max = axes.get_ylim()[1]
    #     axes.set_xlim(axes.get_xlim()[0] - 1, axes.get_xlim()[1] + 1)
    #     for i in range(0, labels.shape[0] - 1):
    #         plt.axvspan(labels.Time[i], labels.Time[i + 1], alpha=0.2, color=labels.Color[i],
    #                     linestyle='-.')
    #         plt.grid()
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
