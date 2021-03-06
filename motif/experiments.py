import config as cfg
import explots
import exputils
import fileutils
import metrics
import motifutils as motif
import visutils as vis
import visualizations


# Runs the segmentation experiment on an audio file with the given name in bin/test.
def segmentation_experiment(name, in_dir, out_dir, write_motifs=False, show_plot=()):
    methods = ('Regular', 'Onset', 'Beat')
    _experiment('Segmentation', name, in_dir, out_dir, methods, write_motifs, show_plot=show_plot)


# Runs the similarity measure experiment on an audio file with the given name in bin/test.
def similarity_experiment(name, in_dir, out_dir, write_motifs=False, show_plot=()):
    methods = ('Match', 'Shazam')
    _experiment('Similarity', name, in_dir, out_dir, methods, write_motifs, show_plot=show_plot)


# Runs the k-means experiment on an audio file with the given name in bin/test.
def k_means_experiment(name, in_dir, out_dir, write_motifs=False, show_plot=()):
    methods = (3, 5, 10)
    _experiment('K-Means', name, in_dir, out_dir, methods, write_motifs, show_plot=show_plot)


# Runs the clustering experiment on an audio file with the given name in bin/test.
def clustering_experiment(name, in_dir, out_dir, write_motifs=False, show_plot=()):
    methods = ('K-Means', 'Spectral', 'Agglom')
    _experiment('Clustering', name, in_dir, out_dir, methods, write_motifs, show_plot=show_plot)


# Generic method for running an experiment. Runs analysis uses an experiment-specific configuration.
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
    if exp_name == 'K-Means':
        lp = 'k='
    else:
        lp = ''

    # Plot the recall, precision, f-measure, boundary measure, and edit distance as bar plots.
    if 'bar' in show_plot:
        fig = vis.get_fig()
        ax = fig.add_subplot(1, 1, 1)
        ax = explots.draw_results_rpf(methods, metric_dict, label_prefix=lp, ax=ax)
        fig.suptitle('{exp_name} Performance for {audio_name}'.format(exp_name=exp_name,
                                                                     audio_name=audio_name))
        vis.save_fig(fig, './bin/graphs/', 'RPF_{}_{}'.format(audio_name, exp_name))

        fig = vis.get_fig()
        explots.draw_results_bed(methods, metric_dict, audio_name, exp_name, fig=fig)
        fig.suptitle("{exp_name} Accuracy on {audio_name}".format(exp_name=exp_name, audio_name=audio_name),
                     fontsize=24)
        if exp_name == 'K-Means':
            ax = fig.get_axes()[0]
            ax.set_xlabel('Number of clusters')
            ax = fig.get_axes()[1]
            ax.set_xlabel('Number of clusters')
        vis.save_fig(fig, './bin/graphs/', 'BED_{}_{}'.format(audio_name, exp_name))

    # Plot the motif segmentations as subplots in a larger figure
    if 'group' in show_plot:
        label_key = 'Ideal'
        methods_grp = (label_key,) + methods
        results[label_key] = ref_motifs
        fig = visualizations.draw_motif_group(audio, fs, results, methods_grp, title='', subplots=(2, 2),
                                              label_prefix=lp)
        fig.suptitle('{exp_name} Motifs on {audio_name}'.format(exp_name=exp_name, audio_name=audio_name))
        vis.save_fig(fig, './bin/graphs/', 'GRP_{}_{}'.format(audio_name, exp_name))

        if exp_name == 'K-Means':
            ax = fig.get_axes()[1]
            ax.set_title(label_key, fontsize=18)

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
    name = 'Avril'
    in_dir = "./bin/test"
    out_dir = "./bin/results"
    segmentation_experiment(name, in_dir, out_dir, show_plot=('group',), write_motifs=False)
    k_means_experiment(name, in_dir, out_dir, show_plot=('group',), write_motifs=False)
    similarity_experiment(name, in_dir, out_dir, show_plot=('group',), write_motifs=False)
    clustering_experiment(name, in_dir, out_dir, show_plot=('group',), write_motifs=False)
    vis.show()
    return


if __name__ == '__main__':
    main()
