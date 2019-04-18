import numpy as np
from matplotlib.ticker import MaxNLocator

import config as cfg
import graphutils as graph
import motifutils as motif
import visutils as vis


def draw_results_reference(audio, fs, labels_df, name='audio', show_plot=None, title_hook='hand-labelled'):
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
        if len(G.nodes) > 20:
            with_labels = False
        else:
            with_labels = True
        draw_results_arc(G, name,
                         title_hook=title_hook,
                         num_groups=num_groups,
                         draw_ref=draw_ref,
                         with_labels=with_labels)
    if 'matrix' in show_plot:
        draw_results_matrix(G, name, title_hook=title_hook)


def draw_results_chord(G, name, title_hook, draw_ref=False):
    chord_labels = graph.to_node_dataframe(G)
    c = vis.plot_chordgraph(G,
                            node_data=chord_labels,
                            label_col=cfg.NODE_LABEL,
                            title='Chord Graph for {} {}'.format(name, title_hook),
                            node_color=cfg.CLUSTER_NAME,
                            edge_color=cfg.CLUSTER_EDGE_NAME)
    vis.show(c)

    if draw_ref:
        c = vis.plot_chordgraph(G,
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

    ax = vis.plot_arcgraph(G,
                           node_order=range(0, num_nodes),
                           node_labels=arc_labels,
                           node_color=group_color,
                           font_size=16,
                           ax=ax
                           )
    ax.set_title('Arc Graph for {} {}'.format(name, title_hook))

    if draw_ref:
        ax = vis.plot_arcgraph(G,
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
    ax = vis.get_axes(ax)
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
    ax.set_ylim(0, 1.0)
    return ax


def draw_results_single(methods, metric_dict, idx, ax=None):
    ax = vis.get_axes(ax)
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


def draw_results_bed(methods, metric_dict, audio_name, exp_name, fig):
    ax = fig.add_subplot(1, 2, 1)
    ax = draw_results_single(methods, metric_dict, 4, ax=ax)
    ax.set_title('Boundary Measure', fontsize=18)
    ax.set_xlabel('Method', fontsize=16)
    ax.set_ylabel('Value (Smaller is Better)', fontsize=16)
    ax.set_ylim(0, 0.50)

    ax = fig.add_subplot(1, 2, 2)
    ax = draw_results_single(methods, metric_dict, 0, ax=ax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Edit Distance', fontsize=18)
    ax.set_xlabel('Method', fontsize=16)
    ax.set_ylabel('Edit Distance (Smaller is Better)', fontsize=16)
    ax.set_ylim(0, 30)

    fig.subplots_adjust(wspace=0.5, top=0.85)
    return


def main():
    return


if __name__ == '__main__':
    main()
