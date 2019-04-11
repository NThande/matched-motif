import numpy as np

import analyzer
import config as cfg
import fileutils
import graphutils as graph
import motifutils as motif
import visutils as vis
import time


def _analysis(analysis_name, audio, fs, length, methods, name='audio', show_plot=(),
              k=cfg.N_ClUSTERS, title_hook='{}'):
    G_dict = {}
    results_dict = {}

    for m in methods:
        if analysis_name is 'Segmentation':
            starts, ends, motif_labels, G = analyzer.analyze(audio, fs,
                                                             num_motifs=k,
                                                             seg_length=length,
                                                             seg_method=m)

        elif analysis_name == 'Similarity':
            starts, ends, motif_labels, G = analyzer.analyze(audio, fs, k,
                                                             seg_length=length,
                                                             similarity_method=m)

        elif analysis_name == 'K-Means':
            starts, ends, motif_labels, G = analyzer.analyze(audio, fs, m,
                                                             seg_length=length,
                                                             cluster_method='kmeans')

        elif analysis_name == 'Clustering':
            starts, ends, motif_labels, G = analyzer.analyze(audio, fs, k,
                                                             seg_length=length,
                                                             cluster_method=m)
        else:
            print("Unrecognized analysis name: {exp_name}".format(exp_name=analysis_name))
            return None, None

        G_dict[m] = G
        results = motif.pack_motif(starts, ends, motif_labels)
        results_dict[m] = results

        title_suffix = title_hook.format(m)
        draw_results(audio, fs, results, show_plot,
                     G=G,
                     name=name,
                     title_hook=title_suffix,
                     draw_ref=(m is methods[0]))
    return results_dict, G_dict


def segmentation_analysis(audio, fs, length, name='audio', show_plot=(),
                          methods=('regular', 'onset'), k=cfg.N_ClUSTERS):
    results_dict, G_dict = _analysis('Segmentation', audio, fs, length, methods,
                                     name=name, show_plot=show_plot, k=k,
                                     title_hook='with {} segmentation')
    return results_dict, G_dict


def similarity_analysis(audio, fs, length, name='audio', show_plot=(),
                        methods=('match', 'shazam'), k=cfg.N_ClUSTERS):
    results_dict, G_dict = _analysis('Similarity', audio, fs, length, methods,
                                     name=name, show_plot=show_plot, k=k,
                                     title_hook='with {} similarity')
    return results_dict, G_dict


def k_means_analysis(audio, fs, length, name='audio', show_plot=(),
                     k_clusters=(cfg.N_ClUSTERS, cfg.N_ClUSTERS + 1)):
    results_dict, G_dict = _analysis('K-Means', audio, fs, length, methods=k_clusters,
                                     name=name, show_plot=show_plot,
                                     title_hook='with {}-means clustering')
    return results_dict, G_dict


def clustering_analysis(audio, fs, length, name='audio', show_plot=(),
                        methods=('kmeans', 'agglom', 'spectral'), k=cfg.N_ClUSTERS):
    results_dict, G_dict = _analysis('Clustering', audio, fs, length, methods,
                                     name=name, show_plot=show_plot, k=k,
                                     title_hook='with {} clustering')
    return results_dict, G_dict


def draw_reference(audio, fs, labels_df, name='audio', show_plot=None, title_hook='hand-labelled'):
    G = graph.from_pandas_labels(labels_df)
    Gp, _ = graph.prune_graph(G)
    num_nodes = len(Gp.nodes())
    group_color = np.zeros(num_nodes)
    for i in Gp.nodes():
        group_color[i] = Gp.nodes()[i][cfg.CLUSTER_NAME]

    title_hook = title_hook
    draw_results_as_network(Gp, name, title_hook=title_hook, show_plot=show_plot)

    if 'motif' in show_plot:
        starts, ends, motif_labels = motif.df_to_motif(labels_df)
        draw_results_motif(audio, fs, starts, ends, motif_labels,
                           name=name,
                           title_hook=title_hook)

    return G


# Master function for drawing qualitative results.
def draw_results(audio, fs, results, show_plot,
                 G=None,
                 name='audio',
                 title_hook='',
                 num_groups=None,
                 draw_ref=False):
    if G is not None:
        draw_results_as_network(G, name,
                                title_hook=title_hook,
                                show_plot=show_plot,
                                num_groups=num_groups,
                                draw_ref=draw_ref)
    if 'motif' in show_plot:
        starts, ends, motif_labels = motif.unpack_motif(results)
        draw_results_motif(audio, fs, starts, ends, motif_labels,
                           name=name,
                           title_hook=title_hook)
    return


def draw_results_as_network(G, name, title_hook, show_plot, num_groups=None, draw_ref=False):
    if show_plot is None:
        return
    if 'chord' in show_plot:
        draw_results_chord(G, name, title_hook=title_hook, draw_ref=draw_ref)
    if 'arc' in show_plot:
        draw_results_arc(G, name, title_hook=title_hook, num_groups=num_groups, draw_ref=draw_ref)
    if 'matrix' in show_plot:
        draw_results_matrix(G, name, title_hook=title_hook)


def draw_results_chord(G, name, title_hook, draw_ref=False):
    chord_labels = graph.to_node_dataframe(G)
    c = vis.draw_chordgraph(G,
                            node_data=chord_labels,
                            label_col=cfg.NODE_LABEL,
                            title='Chord Graph for {} {}'.format(name, title_hook),
                            node_color=cfg.CLUSTER_NAME,
                            edge_color=cfg.CLUSTER_EDGE_NAME)
    vis.show(c)

    if draw_ref:
        c = vis.draw_chordgraph(G,
                                node_data=chord_labels,
                                label_col=cfg.NODE_LABEL,
                                title='Chord Graph for {}'.format(name))
        vis.show(c)


def draw_results_arc(G, name, title_hook, num_groups=None, draw_ref=False):
    # Define group color for arc graph
    arc_labels = graph.to_node_dict(G, node_attr=cfg.NODE_LABEL)
    num_nodes = len(G.nodes())

    if num_groups is not None:
        group_color = np.zeros(num_nodes)
        for i in G.nodes():
            group_color[i] = G.nodes()[i][cfg.CLUSTER_NAME] / num_groups
    else:
        group_color = 'w'

    ax = vis.draw_arcgraph(G,
                           node_size=30.,
                           node_order=range(0, num_nodes),
                           node_labels=arc_labels,
                           node_color=group_color
                           )
    ax.set_title('Arc Graph for {} {}'.format(name, title_hook))

    if draw_ref:
        ax = vis.draw_arcgraph(G,
                               node_size=30.,
                               node_order=range(0, num_nodes),
                               node_labels=arc_labels
                               )
        ax.set_title('Arc Graph for {}'.format(name))


def draw_results_matrix(G, name, title_hook):
    adjacency = graph.graph_to_adjacency_matrix(G)
    ax = vis.plot_similarity_matrix(adjacency)
    ax.set_title("Self-Similarity Matrix for {} {}".format(name, title_hook))


def draw_results_motif(audio, fs, starts, ends, labels, name='audio', title_hook=''):
    ax = vis.plot_motif_segmentation(audio, fs, starts, ends, labels)
    ax.set_title("Motif Segmentation for {} {}".format(name, title_hook))


def draw_results_rpf(methods, metric_dict):
    num_methods = len(methods)
    measures = ('Recall', 'Precision', 'F_measure')
    num_measures = len(measures)

    bar_pos = np.arange(num_measures)
    cur_pos = np.copy(bar_pos)
    bar_width = (0.7 / num_methods)

    ax = None
    color_idx = 0
    for m in methods:
        ax = vis.plot_metric_bar(cur_pos, metric_dict[m][1:4], ax=ax,
                                 group_label=m,
                                 color='C{}'.format(color_idx % 10),
                                 width=bar_width)
        cur_pos = cur_pos + bar_width
        color_idx += 1

    ax.legend()
    ax.set_xticks(bar_pos + ((num_methods - 1) * bar_width) / 2)
    ax.set_xticklabels(measures)
    ax.set_ylabel('Measure')
    return ax


# Write out an entire results set
def write_results(audio, fs, name, out_dir, methods, results):
    for m in methods:
        obs_motifs = results[m]
        write_name = name + ' ' + m
        write_motifs(audio, fs, write_name, out_dir, obs_motifs)


# Write out all identified motifs
def write_motifs(audio, fs, name, audio_dir, motifs):
    id = int(round(time.time() * 1000))
    motif_labels = motifs[cfg.LABEL_IDX]
    num_motifs = motif_labels.shape[0]
    motif_dict = dict.fromkeys(np.unique(motif_labels), 0)
    for i in range(num_motifs):
        motif_start = int(motifs[cfg.START_IDX, i] * fs)
        motif_end = int(motifs[cfg.END_IDX, i] * fs)
        this_motif = audio[motif_start:motif_end]

        this_instance = motif_dict[motif_labels[i]]
        motif_dict[motif_labels[i]] = motif_dict[motif_labels[i]] + 1
        this_name = "{name}_m{motif}_i{instance}_{id}".format(name=name,
                                                              motif=int(motif_labels[i]),
                                                              instance=this_instance,
                                                              id=id)

        fileutils.write_audio(this_motif, fs, this_name, audio_dir)
    return


def tune_length_with_audio(audio, fs):
    return cfg.SEGMENT_LENGTH * np.ceil(((audio.shape[0] / fs) / 30)).astype(int)


def main():
    name = 't4_train'
    in_dir = './bin/labelled'
    out_dir = './bin/results'
    audio, fs = fileutils.load_audio(name, audio_dir=in_dir)
    audio_labels = fileutils.load_labels(name, label_dir=in_dir)
    # Should be sensitive to the length of the track, as well as k
    # Perhaps length should be extended as song goes longer than 30 seconds;
    # 3 second = 30 seconds, 18 seconds = 3 min
    length = tune_length_with_audio(audio, fs)

    draw_reference(audio, fs, audio_labels, name=name,
                   show_plot=('motif'))
    # segmentation_experiment(audio, fs, length, num_motifs=3, name=name,
    #                         show_plot=('arc'))
    # k_means_experiment(audio, fs, length, name=name,
    #                    show_plot=('motif'))
    results, _ = similarity_analysis(audio, fs, length, name=name,
                                     show_plot=('motif'),
                                     methods=('match'), k=cfg.N_ClUSTERS)
    write_motifs(audio, fs, name, out_dir, results['match'])
    vis.show()
    return


if __name__ == '__main__':
    main()
