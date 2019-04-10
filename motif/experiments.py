import fileutils
import exputils
import metrics
import visutils as vis
import motifutils as motif


# The experiments run to generate our output data
def main():
    name = 'genre_test_3'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    audio_labels = fileutils.load_labels(name, label_dir=directory)
    ref_starts, ref_ends, ref_labels = motif.df_to_motif(audio_labels)
    ref_motifs = motif.pack_motif(ref_starts, ref_ends, ref_labels)
    length = 3

    methods = ('match', 'shazam')
    results, _ = exputils.thumbnail_experiment(audio, fs, length, name,
                                               methods=methods, k=5)
    for m in methods:
        obs_motifs = results[m]
        _, _, obs_labels = motif.unpack_motif(obs_motifs)
        this_edit = metrics.edit_distance(obs_labels, ref_labels)
        this_f = metrics.f_measure(obs_motifs, ref_motifs)
        this_bm = metrics.boundary_distance(obs_motifs, ref_motifs)
        print("{} Edit Distance: {}".format(m, this_edit))
        print("{} F Measure: {}".format(m, this_f))
        print("{} Boundary Measure: {}".format(m, this_bm))

    return


if __name__ == '__main__':
    main()
