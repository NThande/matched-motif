import fileutils
from exputils import (thumbnail_experiment, k_means_experiment, segmentation_experiment, draw_reference)
import metrics


# The experiments run to generate our output data
def main():
    name = 'genre_test_3'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    audio_labels = fileutils.load_labels(name, label_dir=directory)
    length = 3

    draw_reference(audio, fs, audio_labels, name=name,
                   show_plot=())
    segmentation_experiment(audio, fs, length, num_motifs=3, name=name,
                            show_plot=())
    k_means_experiment(audio, fs, length, name=name,
                       show_plot=())
    thumbnail_experiment(audio, fs, length, name=name,
                         show_plot=(),
                         methods=('match', 'shazam'), k=3)
    return


if __name__ == '__main__':
    main()
