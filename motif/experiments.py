import fileutils
import visutils as vis
import graphutils as graph
import config as cfg
import analyzer
import numpy as np


def segmentation_experiment(audio, fs, length, num_motifs, name='audio', show_plot=None):
    seg_methods = ['regular', 'onset']
    G_set = []
    # chords = []

    for method in seg_methods:
        _, G = analyzer.analyze(audio, fs, num_motifs, seg_length=length, seg_method=method)
        G_set.append(G)

        title_hook = 'with {} segmentation'.format(method)
        draw_results(G, name,
                     title_hook=title_hook,
                     show_plot=show_plot)

        # if 'chord' in show_plot:
        #     chord_labels = graph.to_node_dataframe(G)
        #     c = vis.draw_chordgraph(G,
        #                             node_data=chord_labels,
        #                             label_col=cfg.NODE_LABEL,
        #                             title='Chord Graph of {} with {} segmentation'.format(name, method))
        #     vis.show(c)
        #     # chords.append(c)
        #
        # if 'arc' in show_plot:
        #     arc_labels = graph.to_node_dict(G, node_attr=cfg.NODE_LABEL)
        #     ax = vis.draw_arcgraph(G,
        #                            node_size=30.,
        #                            node_labels=arc_labels,
        #                            node_order=range(0, len(G.nodes())))
        #     ax.set_title("Arc Graph of {} with {} segmentation".format(name, method))
        #
        # if 'matrix' in show_plot:
        #     adjacency = graph.graph_to_adjacency_matrix(G)
        #     ax = vis.plot_similarity_matrix(adjacency)
        #     ax.set_title("Self-Similarity Matrix for {} with {} segmentation".format(name, method))

    # if 'chord' in show_plot:
    #     for c in chords:
    #         vis.show(c)
    return G_set


def k_means_experiment(audio, fs, length, name='audio', show_plot=None,
                       k_clusters=range(2, 7)):
    G_set = []
    # chords = []
    for k in k_clusters:
        motifs, G = analyzer.analyze(audio, fs, k, seg_length=length, cluster_method='kmeans')
        G_set.append(G)

        title_hook = 'with {}-means clustering'.format(k)
        draw_results(G, name,
                     title_hook=title_hook,
                     show_plot=show_plot,
                     num_groups=len(motifs),
                     draw_base= (k == k_clusters[0]))

        # if 'chord' in show_plot:
        #     # Add chord without cluster coloring for comparision
        #     chord_labels = graph.to_node_dataframe(G)
        #     if k == k_clusters[0]:
        #         c = vis.draw_chordgraph(G,
        #                                 node_data=chord_labels,
        #                                 label_col=cfg.NODE_LABEL,
        #                                 title='Chord Graph for {}'.format(name))
        #         vis.show(c)
        #         # chords.append(c)
        #
        #     c = vis.draw_chordgraph(G,
        #                             node_data=chord_labels,
        #                             label_col=cfg.NODE_LABEL,
        #                             title='Chord Graph for {} with {}-means clustering'.format(name, k),
        #                             node_color=cfg.CLUSTER_NAME,
        #                             edge_color=cfg.CLUSTER_EDGE_NAME)
        #     vis.show(c)
        #     # chords.append(c)
        #
        # if 'arc' in show_plot:
        #     # Define group color for arc graph
        #     arc_labels = graph.to_node_dict(G, node_attr=cfg.NODE_LABEL)
        #     num_nodes = len(G.nodes())
        #     group_color = np.zeros(num_nodes)
        #     for i in G.nodes():
        #         group_color[i] = G.nodes()[i][cfg.CLUSTER_NAME] / len(motifs)
        #
        #     # Add basic arc graph for comparision
        #     if k == k_clusters[0]:
        #         ax = vis.draw_arcgraph(G,
        #                                node_size=30.,
        #                                node_order=range(0, num_nodes),
        #                                node_labels=arc_labels
        #                                )
        #         ax.set_title('Arc Graph for {}'.format(name))
        #
        #     ax = vis.draw_arcgraph(G,
        #                            node_size=30.,
        #                            node_order=range(0, num_nodes),
        #                            node_labels=arc_labels,
        #                            node_color=group_color
        #                            )
        #     ax.set_title('Arc Graph for {} with {}-means clustering'.format(name, k))

    # if 'chord' in show_plot:
    #     for c in chords:
    #         vis.show(c)
    return G_set


def draw_reference(audio_df, name='audio', show_plot=None):
    G = graph.from_pandas_labels(audio_df)
    Gp = graph.prune_graph(G)
    num_nodes = len(G.nodes())
    group_color = np.zeros(num_nodes)
    for i in G.nodes():
        group_color[i] = G.nodes()[i][cfg.CLUSTER_NAME]

    title_hook = 'hand-labelled'
    draw_results(Gp, name, title_hook=title_hook, show_plot=show_plot)

    # if 'chord' in show_plot:
    #     chord_labels = graph.to_node_dataframe(Gp)
    #     c = vis.draw_chordgraph(Gp,
    #                             node_data=chord_labels,
    #                             label_col=cfg.NODE_LABEL,
    #                             title='Reference Chord Graph of {}'.format(name),
    #                             node_color=cfg.CLUSTER_NAME,
    #                             edge_color=cfg.CLUSTER_EDGE_NAME
    #                             )
    #     vis.show(c)
    #
    # if 'arc' in show_plot:
    #     arc_labels = graph.to_node_dict(G, node_attr=cfg.NODE_LABEL)
    #     ax = vis.draw_arcgraph(G,
    #                            node_size=30.,
    #                            node_labels=arc_labels,
    #                            node_order=range(0, len(G.nodes())),
    #                            node_color=group_color
    #                            )
    #     ax.set_title("Reference Arc Graph of {}".format(name))
    #
    # if 'matrix' in show_plot:
    #     adjacency = graph.graph_to_adjacency_matrix(G)
    #     ax = vis.plot_similarity_matrix(adjacency)
    #     ax.set_title("Reference Self-Similarity Matrix for {}".format(name))

    return G


def draw_results(G, name, title_hook, show_plot, num_groups=None, draw_base=False):
    if 'chord' in show_plot:
        draw_results_chord(G, name, title_hook=title_hook, draw_base=draw_base)
    if 'arc' in show_plot:
        draw_results_arc(G, name, title_hook=title_hook, num_groups=num_groups, draw_base=draw_base)
    if 'matrix' in show_plot:
        draw_results_matrix(G, name, title_hook=title_hook)


def draw_results_chord(G, name, title_hook, draw_base=False):
    chord_labels = graph.to_node_dataframe(G)
    c = vis.draw_chordgraph(G,
                            node_data=chord_labels,
                            label_col=cfg.NODE_LABEL,
                            title='Chord Graph for {} {}'.format(name, title_hook),
                            node_color=cfg.CLUSTER_NAME,
                            edge_color=cfg.CLUSTER_EDGE_NAME)
    vis.show(c)

    if draw_base:
        c = vis.draw_chordgraph(G,
                                node_data=chord_labels,
                                label_col=cfg.NODE_LABEL,
                                title='Chord Graph for {}'.format(name))
        vis.show(c)


def draw_results_arc(G, name, title_hook, num_groups=None, draw_base=False):
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

    if draw_base:
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


def main():
    name = 't3_train'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    audio_labels = fileutils.load_labels(name, label_dir=directory)
    length = 3

    draw_reference(audio_labels, name=name, show_plot=('chord', 'arc'))
    segmentation_experiment(audio, fs, length, num_motifs=3, name=name, show_plot=('chord', 'arc'))
    k_means_experiment(audio, fs, length, name=name, show_plot=('chord', 'arc'))
    vis.show()
    return


if __name__ == '__main__':
    main()
