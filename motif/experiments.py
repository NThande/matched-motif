import config as cfg
import exputils
import fileutils
import metrics
import motifutils as motif
import visutils as vis


def segmentation_experiment(name, in_dir, out_dir, write_motifs=False):
    methods = ('regular', 'onset', 'beat')
    _experiment('Segmentation', name, in_dir, out_dir, methods, write_motifs)


def similarity_experiment(name, in_dir, out_dir, write_motifs=False):
    methods = ('match', 'shazam')
    _experiment('Similarity', name, in_dir, out_dir, methods, write_motifs)


def k_means_experiment(name, in_dir, out_dir, write_motifs=False):
    methods = (3, 5, 10)
    _experiment('K-Means', name, in_dir, out_dir, methods, write_motifs)


def clustering_experiment(name, in_dir, out_dir, write_motifs=False):
    methods = ('kmeans', 'spectral', 'agglom')
    _experiment('Clustering', name, in_dir, out_dir, methods, write_motifs)


def _experiment(exp_name, audio_name, in_dir, out_dir,
                methods, write_motifs=False):
    audio, fs = fileutils.load_audio(audio_name, audio_dir=in_dir)

    audio_labels = fileutils.load_labels(audio_name, label_dir=in_dir)
    ref_starts, ref_ends, ref_labels = motif.df_to_motif(audio_labels)
    ref_motifs = motif.pack_motif(ref_starts, ref_ends, ref_labels)

    length = cfg.SEGMENT_LENGTH

    if exp_name == 'Segmentation':
        results, _ = exputils.segmentation_analysis(audio, fs, length, audio_name,
                                                    methods=methods, k=cfg.N_ClUSTERS)
    elif exp_name == 'Similarity':
        results, _ = exputils.similarity_analysis(audio, fs, length, audio_name,
                                                  methods=methods, k=cfg.N_ClUSTERS)
    elif exp_name == 'K-Means':
        results, _ = exputils.k_means_analysis(audio, fs, length, audio_name,
                                               k_clusters=methods)

    elif exp_name == 'Clustering':
        results, _ = exputils.clustering_analysis(audio, fs, length, audio_name,
                                                  methods=methods, k=cfg.N_ClUSTERS)
    else:
        print("Unrecognized experiment name: {exp_name}".format(exp_name=exp_name))
        return

    metric_dict = results_to_metrics(results, methods, ref_motifs)

    ax = exputils.draw_results_rpf(methods, metric_dict)
    ax.set_title('{exp_name} Comparision for {audio_name}'.format(exp_name=exp_name,
                                                                  audio_name=audio_name))

    if write_motifs:
        exputils.write_results(audio, fs, audio_name, out_dir, methods, results)


# Convert results dict to metrics dict
def results_to_metrics(results, methods, ref_motifs):
    _, _, ref_labels = motif.unpack_motif(ref_motifs)
    metric_dict = dict.fromkeys(methods)

    for m in methods:
        obs_motifs = results[m]
        _, _, obs_labels = motif.unpack_motif(obs_motifs)

        this_edit = metrics.edit_distance(obs_labels, ref_labels)
        this_recall = metrics.recall(obs_motifs, ref_motifs)
        this_precis = metrics.precision(obs_motifs, ref_motifs)
        this_f = metrics.f_measure(obs_motifs, ref_motifs)
        this_bm = metrics.boundary_distance(obs_motifs, ref_motifs)
        metric_dict[m] = [this_edit, this_recall, this_precis, this_f, this_bm]

    return metric_dict


# The experiments run to generate our output data
def main():
    name = 't3_train'
    in_dir = "./bin/test"
    out_dir = "./bin/results"
    segmentation_experiment(name, in_dir, out_dir)
    k_means_experiment(name, in_dir, out_dir)
    similarity_experiment(name, in_dir, out_dir)
    clustering_experiment(name, in_dir, out_dir)
    vis.show()
    return


if __name__ == '__main__':
    main()
