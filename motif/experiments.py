
import config as cfg
import explots
import exputils
import fileutils
import metrics
import motifutils as motif
import visutils as vis
import visualizations


def segmentation_experiment(name, in_dir, out_dir, write_motifs=False, show_plot=()):
    methods = ('Regular', 'Onset', 'Beat')
    _experiment('Segmentation', name, in_dir, out_dir, methods, write_motifs, show_plot=show_plot)


def similarity_experiment(name, in_dir, out_dir, write_motifs=False, show_plot=()):
    methods = ('Match', 'Shazam')
    _experiment('Similarity', name, in_dir, out_dir, methods, write_motifs, show_plot=show_plot)


def k_means_experiment(name, in_dir, out_dir, write_motifs=False, show_plot=()):
    methods = (3, 5, 10)
    _experiment('K-Means', name, in_dir, out_dir, methods, write_motifs, show_plot=show_plot)


def clustering_experiment(name, in_dir, out_dir, write_motifs=False, show_plot=()):
    methods = ('K-Means', 'Spectral', 'Agglom')
    _experiment('Clustering', name, in_dir, out_dir, methods, write_motifs, show_plot=show_plot)


def _experiment(exp_name, audio_name, in_dir, out_dir,
                methods, write_motifs=False, show_plot=()):
    audio, fs = fileutils.load_audio(audio_name, audio_dir=in_dir)

    audio_labels = fileutils.load_labels(audio_name, label_dir=in_dir)
    ref_starts, ref_ends, ref_labels = motif.df_to_motif(audio_labels)
    ref_motifs = motif.pack_motif(ref_starts, ref_ends, ref_labels)

    length = cfg.SEGMENT_LENGTH

    if exp_name == 'Segmentation':
        results, _ = exputils.segmentation_analysis(audio, fs, length, audio_name,
                                                    methods=methods, k=cfg.N_ClUSTERS,
                                                    show_plot=show_plot)
    elif exp_name == 'Similarity':
        results, _ = exputils.similarity_analysis(audio, fs, length, audio_name,
                                                  methods=methods, k=cfg.N_ClUSTERS,
                                                  show_plot=show_plot)
    elif exp_name == 'K-Means':
        results, _ = exputils.k_means_analysis(audio, fs, length, audio_name,
                                               k_clusters=methods,
                                               show_plot=show_plot)

    elif exp_name == 'Clustering':
        results, _ = exputils.clustering_analysis(audio, fs, length, audio_name,
                                                  methods=methods, k=cfg.N_ClUSTERS,
                                                  show_plot=show_plot)
    else:
        print("Unrecognized experiment name: {exp_name}".format(exp_name=exp_name))
        return

    metric_dict = results_to_metrics(results, methods, ref_motifs)

    # Output Plots
    if 'bar' in show_plot:
        if exp_name == 'K-Means':
            lp = 'k='
        else:
            lp = ''
        fig, ax = explots.draw_results_rpf(methods, metric_dict, label_prefix=lp)
        fig.suptitle('{exp_name} Comparison for {audio_name}'.format(exp_name=exp_name,
                                                                     audio_name=audio_name))

        fig = explots.draw_results_bed(methods, metric_dict, audio_name, exp_name)
        fig.suptitle("{exp_name} Experiment on {audio_name}".format(exp_name=exp_name, audio_name=audio_name),
                     fontsize=24)
        vis.save_fig(fig, './bin/graphs/', 'BED_{}_{}'.format(audio_name, exp_name))

    if 'group' in show_plot:
        label_key = 'Ideal'
        methods_grp = (label_key,) + methods
        results[label_key] = ref_motifs
        fig = visualizations.draw_motif_group(audio, fs, results, methods_grp, title='', subplots=(2, 2))
        fig.suptitle('{exp_name} Motifs on {audio_name}'.format(exp_name=exp_name, audio_name=audio_name))
        vis.save_fig(fig, './bin/graphs/', 'GRP_{}_{}'.format(audio_name, exp_name))

    if write_motifs:
        exputils.write_results(audio, fs, audio_name, out_dir, methods, results)

    return metric_dict


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
    name = 't1'
    in_dir = "./bin/test"
    out_dir = "./bin/results"
    segmentation_experiment(name, in_dir, out_dir, show_plot=('bar', 'group'), write_motifs=False)
    # k_means_experiment(name, in_dir, out_dir, show_plot=('matrix', 'motif'), write_motifs=False)
    # similarity_experiment(name, in_dir, out_dir, show_plot=(), write_motifs=False)
    # clustering_experiment(name, in_dir, out_dir, show_plot=('arc',), write_motifs=False)
    vis.show()
    return


if __name__ == '__main__':
    main()
