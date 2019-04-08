import fileutils
import visutils as vis
import graphutils as graph
import config as cfg
import analyzer
import numpy as np


def segmentation_experiment(audio, fs, length, num_motifs, name='audio', show_plot=None):
    seg_methods = ['regular', 'onset']
    G_set = []

    for method in seg_methods:
        starts, ends, motif_labels, G = analyzer.analyze(audio, fs, num_motifs, seg_length=length, seg_method=method)
        G_set.append(G)

        title_hook = 'with {} segmentation'.format(method)
        if G is not None:
            draw_results_as_network(G, name,
                                    title_hook=title_hook,
                                    show_plot=show_plot)
    return G_set


def k_means_experiment(audio, fs, length, name='audio', show_plot=None,
                       k_clusters=(3, 10)):
    G_set = []
    for k in k_clusters:
        starts, ends, motif_labels, G = analyzer.analyze(audio, fs, k,
                                                         seg_length=length,
                                                         seg_method='onset',
                                                         cluster_method='kmeans')
        G_set.append(G)
        num_motifs = np.unique(motif_labels).shape[0]

        title_hook = 'with {}-means clustering'.format(k)
        if G is not None:
            draw_results_as_network(G, name,
                                    title_hook=title_hook,
                                    show_plot=show_plot,
                                    num_groups=num_motifs,
                                    draw_ref=(k == k_clusters[0]))
        if 'motif' in show_plot:
            draw_results_motif(audio, fs, starts, ends, motif_labels,
                               name=name,
                               title_hook=title_hook)
    return G_set


def thumbnail_experiment(audio, fs, length, name='audio', show_plot=None,
                         methods=('match', 'shazam'), k=3):
    G_set = []
    # If only one method string is passed in
    if not isinstance(methods, tuple):
        methods = [methods]

    for m in methods:
        starts, ends, motif_labels, G = analyzer.analyze(audio, fs, k,
                                                         seg_length=length,
                                                         similarity_method=m)
        G_set.append(G)
        num_motifs = np.unique(motif_labels).shape[0]

        title_hook = 'with {} similarity'.format(m)
        if G is not None:
            draw_results_as_network(G, name,
                                    title_hook=title_hook,
                                    show_plot=show_plot,
                                    num_groups=num_motifs,
                                    draw_ref=False)
        if 'motif' in show_plot:
            draw_results_motif(audio, fs, starts, ends, motif_labels,
                               name=name,
                               title_hook=title_hook)
    return G_set


def draw_reference(audio, fs, labels_df, name='audio', show_plot=None):
    G = graph.from_pandas_labels(labels_df)
    Gp = graph.prune_graph(G)
    num_nodes = len(G.nodes())
    group_color = np.zeros(num_nodes)
    for i in G.nodes():
        group_color[i] = G.nodes()[i][cfg.CLUSTER_NAME]

    title_hook = 'hand-labelled'
    draw_results_as_network(Gp, name, title_hook=title_hook, show_plot=show_plot)

    if 'motif' in show_plot:
        labels = labels_df['Event'].values
        segments = labels_df['Time'].values
        num_segments = segments.shape[0]

        starts = segments[:num_segments - 1]
        ends = segments[1:]
        motif_labels = labels[:num_segments - 1]

        draw_results_motif(audio, fs, starts, ends, motif_labels,
                           name=name,
                           title_hook=title_hook)

    return G


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


def main():
    name = 'genre_test_7'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    # audio_labels = fileutils.load_labels(name, label_dir=directory)
    length = 3

    # draw_reference(audio_labels, name=name,
    #                show_plot=('chord', 'arc'))
    # segmentation_experiment(audio, fs, length, num_motifs=3, name=name,
    #                         show_plot=('arc'))
    # k_means_experiment(audio, fs, length, name=name,
    #                    show_plot=('motif'))
    thumbnail_experiment(audio, fs, length, name=name,
                         show_plot=('motif', 'matrix'),
                         methods=('match','shazam'), k=3)
    vis.show()
    return


if __name__ == '__main__':
    main()
