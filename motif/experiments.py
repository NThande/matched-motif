from matplotlib.ticker import MaxNLocator

import config as cfg
import explots
import exputils
import fileutils
import metrics
import motifutils as motif
import visutils as vis


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
    fig = vis.get_fig()
    ax = fig.add_subplot(1, 1, 1)
    if exp_name == 'K-Means':
        lp = 'k='
    else:
        lp = ''
    ax = explots.draw_results_rpf(methods, metric_dict, ax=ax, label_prefix=lp)
    ax.set_title('{exp_name} Comparison for {audio_name}'.format(exp_name=exp_name,
                                                                 audio_name=audio_name))
    fig.savefig("./bin/graphs/" + "RPF_" + audio_name + "_" + exp_name + ".png", dpi=150)

    fig = vis.get_fig()
    fig.set_size_inches(8.0, 5.5)
    ax = fig.add_subplot(1, 2, 1)
    ax = explots.draw_results_single(methods, metric_dict, 4, ax=ax)
    ax.set_title('Boundary Measure', fontsize=18)
    ax.set_xlabel('Method', fontsize=16)
    ax.set_ylabel('Value (Smaller is Better)', fontsize=16)

    fig.subplots_adjust(wspace=0.5, top=0.85)
    ax = fig.add_subplot(1, 2, 2)
    ax = explots.draw_results_single(methods, metric_dict, 0, ax=ax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Edit Distance', fontsize=18)
    ax.set_xlabel('Method', fontsize=16)
    ax.set_ylabel('Edit Distance (Smaller is Better)', fontsize=16)

    fig.suptitle("{exp_name} Experiment on {audio_name}".format(exp_name=exp_name, audio_name=audio_name),
                 fontsize=24)
    fig.savefig("./bin/graphs/" + "BED_" + audio_name + "_" + exp_name + ".png", dpi=150)

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
    name = 't1'
    in_dir = "./bin/test"
    out_dir = "./bin/results"
    segmentation_experiment(name, in_dir, out_dir, show_plot=(), write_motifs=False)
    # k_means_experiment(name, in_dir, out_dir, show_plot=('matrix', 'motif'), write_motifs=False)
    # similarity_experiment(name, in_dir, out_dir, show_plot=(), write_motifs=False)
    # clustering_experiment(name, in_dir, out_dir, show_plot=('arc',), write_motifs=False)
    vis.show()
    return


if __name__ == '__main__':
    main()
