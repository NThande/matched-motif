import librosa.display
import numpy as np

import analyzer
import config as cfg
import graphutils as graph
import match_filter
import motifutils as motif
import visutils as vis


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


def draw_results_arc(G, name, title_hook, num_groups=None, draw_ref=False, ax=None, with_labels=True):
    # Define group color for arc graph
    if with_labels:
        arc_labels = graph.to_node_dict(G, node_attr=cfg.NODE_LABEL)
    else:
        arc_labels = None
    num_nodes = len(G.nodes())

    if num_groups is not None:
        group_color = np.zeros(num_nodes)
        for i in G.nodes():
            group_color[i] = G.nodes()[i][cfg.CLUSTER_NAME] / num_groups
    else:
        group_color = 'w'

    ax = vis.draw_arcgraph(G,
                           node_size=50.,
                           node_order=range(0, num_nodes),
                           node_labels=arc_labels,
                           node_color=group_color,
                           font_size=16,
                           ax=ax
                           )
    ax.set_title('Arc Graph for {} {}'.format(name, title_hook))

    if draw_ref:
        ax = vis.draw_arcgraph(G,
                               node_size=20.,
                               node_order=range(0, num_nodes),
                               node_labels=arc_labels,
                               font_size=16
                               )
        ax.set_title('Arc Graph for {}'.format(name))


def draw_results_matrix(G, name, title_hook, ax=None):
    adjacency = graph.graph_to_adjacency_matrix(G)
    ax = vis.plot_similarity_matrix(adjacency, ax)
    ax.set_title("Self-Similarity Matrix for {} {}".format(name, title_hook))
    return ax


def draw_results_motif(audio, fs, starts, ends, labels, name='audio', title_hook='', ax=None):
    ax = vis.plot_motif_segmentation(audio, fs, starts, ends, labels, ax=ax)
    ax.set_title("Motif Segmentation for {} {}".format(name, title_hook), pad=20)


def draw_results_rpf(methods, metric_dict, ax=None, label_prefix=''):
    num_methods = len(methods)
    measures = ('Recall', 'Precision', 'F-Measure')
    num_measures = len(measures)

    bar_pos = np.arange(num_measures)
    cur_pos = np.copy(bar_pos)
    bar_width = (0.7 / num_methods)

    color_idx = 0
    for m in methods:
        ax = vis.plot_metric_bar(cur_pos, metric_dict[m][1:4], ax=ax,
                                 group_label=label_prefix + str(m),
                                 color='C{}'.format(color_idx % 10),
                                 width=bar_width)
        cur_pos = cur_pos + bar_width
        color_idx += 1

    ax.legend(frameon=True)
    ax.set_xticks(bar_pos + ((num_methods - 1) * bar_width) / 2)
    ax.set_xticklabels(measures)
    ax.set_ylabel('Value (0.0 - 1.0)')
    ax.set_xlabel('Performance Measure (Larger is Better)')
    return ax


def draw_results_single(methods, metric_dict, idx, ax=None):
    num_methods = len(methods)
    bar_width = 0.175
    bar_pos = np.arange(num_methods) * bar_width * 2

    pos_idx = 0
    for m in methods:
        ax = vis.plot_metric_bar(bar_pos[pos_idx], metric_dict[m][idx], ax=ax,
                                 width=bar_width, color='C0')
        pos_idx += 1

    ax.set_xticks(bar_pos)
    ax.set_xticklabels(methods)
    return ax


# Some specific plots that I need

def draw_segmentation_evolution(audio, fs, length, name='audio'):
    return


def draw_matrix_evolution(audio, fs, length, name='audio'):
    adjacency_list = []
    title_list = []
    _, _, segments, adjacency = analyzer.self_similarity(audio, fs, length, method='Match', seg_method='Beat')
    adjacency_list.append(adjacency)
    title_list.append('Default')

    adjacency_no_overlap = analyzer.remove_overlap(segments, np.copy(adjacency), length)
    adjacency_list.append(adjacency_no_overlap)
    title_list.append('No Overlap')

    adjacency_reweight = analyzer.reweight_by_time(segments, np.copy(adjacency_no_overlap))
    adjacency_list.append(adjacency_reweight)
    title_list.append('Time Weight Added')

    adjacency_thresh = analyzer.topk_threshold(np.copy(adjacency_no_overlap), 3)
    adjacency_list.append(adjacency_thresh)
    title_list.append('With Threshold')

    fig = vis.get_fig()
    fig.suptitle('Self-Similarity Matrix Evolution')
    num_plots = len(adjacency_list)
    plots_per_row = 2
    plots_per_col = np.ceil(num_plots / plots_per_row)

    ax_over = fig.add_subplot(1, 1, 1)  # The big subplot
    ax_over.set_xlabel('Window #')
    ax_over.set_ylabel('Window #')
    ax_over.spines['top'].set_color('none')
    ax_over.spines['bottom'].set_color('none')
    ax_over.spines['left'].set_color('none')
    ax_over.spines['right'].set_color('none')
    ax_over.tick_params(labelcolor='w', top=False, bottom=True, left=False, right=False)

    for i in range(1, num_plots + 1):
        ax = fig.add_subplot(plots_per_row, plots_per_col, i)
        adj = adjacency_list[i - 1]

        image = ax.imshow(adj)
        ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
        ax.set_title(title_list[i - 1], fontsize=16)

        if i == num_plots:
            fig.subplots_adjust(right=0.8, top=0.85, wspace=0.4, hspace=0.4)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(image, cax=cbar_ax, format='%2.2f')
            cbar.set_label('Similarity')
    # fig = vis.plot_similarity_matrix_group(fig, adjacency_list, title_list)
    vis.save_fig(fig, "./bin/graphs/", "SSM_{}".format(name))
    return


def draw_matrix_arc_chord_demo(G, name):
    fig = vis.get_fig()
    fig.suptitle('Self-Similarity Visualizations')

    ax_over = fig.add_subplot(1, 1, 1)  # The big subplot
    # ax_over.set_xlabel('Window #')
    # ax_over.set_ylabel('Window #')
    ax_over.spines['top'].set_color('none')
    ax_over.spines['bottom'].set_color('none')
    ax_over.spines['left'].set_color('none')
    ax_over.spines['right'].set_color('none')
    ax_over.tick_params(labelcolor='w', top=False, bottom=True, left=False, right=False)

    ax = fig.add_subplot(1, 2, 1)
    adjacency = graph.graph_to_adjacency_matrix(G)
    ax.imshow(adjacency)
    ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
    ax.set_xlabel('Time Windows')
    ax.set_ylabel('Time Windows')
    ax.set_title('Self-Similarity Matrix')
    ax = fig.add_subplot(1, 2, 2)
    arc_labels = graph.to_node_dict(G, node_attr=cfg.NODE_LABEL)
    ax = vis.draw_arcgraph(G,
                           node_size=20.,
                           node_color='k',
                           node_labels=arc_labels,
                           font_size=20,
                           ax=ax,
                           vertical_shift=2)
    ax.set_xlabel("Time Segments")
    ax.set_title('Arc Diagram')
    chord_labels = graph.to_node_dataframe(G)
    c = vis.draw_chordgraph(G,
                            node_data=chord_labels,
                            label_col=cfg.NODE_LABEL,
                            title='Chord Graph Visualization')
    # fig = vis.plot_similarity_matrix_group(fig, adjacency_list, title_list)
    vis.show(c)
    vis.save_fig(fig, "./bin/graphs/", "SSM_{}".format(name))
    return


def draw_motif_group(audio, fs, results, methods, labels_df):
    fig = vis.get_fig()
    plots_per_row = 2
    plots_per_col = 2
    f_idx = 2

    ax = fig.add_subplot(plots_per_row, plots_per_col, 1)
    starts, ends, labels = motif.df_to_motif(labels_df)
    librosa.display.waveplot(audio, fs, ax=ax, color='gray')
    labels = labels.astype(int)
    vis.add_motif_labels(ax, starts, ends, labels, alpha=0.5)
    ax.set_title('Ideal', fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('')

    for m in methods:
        ax = fig.add_subplot(plots_per_col, plots_per_row, f_idx)
        starts, ends, labels = motif.unpack_motif(results[m])
        librosa.display.waveplot(audio, fs, ax=ax, color='gray')
        labels = labels.astype(int)
        vis.add_motif_labels(ax, starts, ends, labels, alpha=0.5)
        ax.set_title(str(m), fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('')

        f_idx += 1
    fig.subplots_adjust(right=0.8, wspace=0.4, hspace=0.4)


def draw_segmentation_comparison(audio, fs):
    fig = vis.get_fig()
    fig.suptitle('Comparison of Segmentation Windows')
    ax_over = fig.add_subplot(1, 1, 1)  # The big subplot
    ax_over.set_xlabel('Time (Seconds)')
    ax_over.spines['top'].set_color('none')
    ax_over.spines['bottom'].set_color('none')
    ax_over.spines['left'].set_color('none')
    ax_over.spines['right'].set_color('none')
    ax_over.tick_params(labelcolor='w', top=False, bottom=True, left=False, right=False)

    ax = fig.add_subplot(2, 2, 1)
    ax.set_title('Audio Waveform', fontsize=14)
    librosa.display.waveplot(audio, fs, ax=ax, color='gray')
    ax.set_xlabel('')
    audio_len = int(audio.shape[0] / fs)
    ax.set_xticks(np.arange(audio_len + 1, step=np.ceil(audio_len / 5)))

    thumb, similarity, segments, sim_matrix = match_filter.thumbnail(audio, fs,
                                                                     seg_method='regular',
                                                                     length=2)

    ax = fig.add_subplot(2, 2, 2)
    ax.set_ylabel('Window #', fontsize=14)
    ax = vis.plot_window_overlap(segments, np.ones(segments.shape) * 2, audio_len, tick_step=2, ax=ax)
    ax.set_title('Regular', fontsize=14)
    ax.grid()

    thumb, similarity, segments, sim_matrix = match_filter.thumbnail(audio, fs,
                                                                     seg_method='onset',
                                                                     length=2)
    ax = fig.add_subplot(2, 2, 3)
    ax.set_ylabel('Window #', fontsize=14)
    ax = vis.plot_window_overlap(segments, np.ones(segments.shape) * 2, audio_len, tick_step=5, ax=ax)
    ax.set_title('Onset', fontsize=14)
    ax.grid()

    thumb, similarity, segments, sim_matrix = match_filter.thumbnail(audio, fs,
                                                                     seg_method='beat',
                                                                     length=2)
    print(segments)
    ax = fig.add_subplot(2, 2, 4)
    ax.set_ylabel('Window #', fontsize=14)
    ax = vis.plot_window_overlap(segments, np.ones(segments.shape) * 2, audio_len, tick_step=3, ax=ax)
    ax.set_title('Beat', fontsize=14)
    ax.grid()

    fig.subplots_adjust(right=0.8, top=0.85, wspace=0.4, hspace=0.4)


def main():
    return


if __name__ == '__main__':
    main()
