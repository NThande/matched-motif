import numpy as np
import fileutils
import fingerprint as fp
import segmentation as seg
import visutils as vis


# Apply a sliding window of a part of the song to the rest of the song
def thumbnail(audio, fs, audio_pairs, window_length=2, overlap=0.5):
    segments = seg.segment_regular(audio, fs, length=window_length, overlap=overlap)
    # print(segments)
    num_segments = segments.shape[0]
    seg_matches = np.zeros(num_segments)

    # Identify similarity of segment to the entire pieece of audio
    for i in range(0, num_segments - 1):
        snap_start = int(segments[i] * fs)
        snap_end = int(segments[i + 1] * fs)
        segment = audio[snap_start: snap_end]
        _, seg_pairs = fp.fingerprint(segment)

        segment_hits = fp.linear_search(audio_pairs, seg_pairs)

        seg_matches[i] = np.average(segment_hits)
        # print("Completed Window {} / {}".format(i + 1, num_segments))

    # Normalize and identify maximum window
    seg_matches = seg_matches / np.max(seg_matches)
    thumb_idx = np.argmax(seg_matches)
    thumb_segment = audio[int(segments[thumb_idx] * fs): int(segments[thumb_idx] * fs)]

    # print("Thumbnail complete")
    return thumb_segment, seg_matches, segments


# Run the thumbnailing strategy for different window lengths
def thumbnail_multi(audio, fs, audio_pairs, length_vect, overlap_vect):
    num_experiments = length_vect.shape[0]
    matches_list = []
    segments_list =[]
    for i in range(0, num_experiments):
        _, matches, segments = thumbnail(audio, fs, audio_pairs,
                                         window_length=length_vect[i],
                                         overlap=overlap_vect[i])
        matches_list.append(matches)
        segments_list.append(segments)
    return matches_list, segments_list


def main():
    name = 't3_train'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    audio_labels = fileutils.load_labels(name, label_dir=directory)

    _, audio_pairs = fp.fingerprint(audio)
    _, matches, segments = thumbnail(audio, fs, audio_pairs)
    ax = vis.plot_similarity_curve(matches, segments, labels=audio_labels)
    ax.set_title('Similarity Curve for {}'.format(name))
    vis.show()


if __name__ == '__main__':
    main()
