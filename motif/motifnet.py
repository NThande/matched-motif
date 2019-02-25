import networkx as nx
import fileutils
import matchfilt as mf
import numpy as np
from sklearn import cluster
import visutils as vis
import graphutils as grp


def cluster_k_means(g, k_clusters, weight='weight', n_init=200):
    onset_mat = nx.incidence_matrix(g, weight=weight)
    kmeans_clf = cluster.KMeans(n_clusters=k_clusters, n_init=n_init)
    kmeans = kmeans_clf.fit_predict(onset_mat)
    return kmeans


def main():
    name = 't3_train'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    length = 3
    k_clusters = 3
    label_col = 'Time'
    node_color_col = 'Group'
    edge_color_col = 'Edge Group'

    # # Regular Segmentation
    # _, _, segments_reg, adj_reg = mf.thumbnail(audio, fs, length=length)
    #
    # # Format segment labels
    # reg_labels = []
    # for i in range(segments_reg.shape[0]):
    #     reg_labels.append('%2.2f - %2.2f' % (segments_reg[i], segments_reg[i] + length))
    #
    # adj_reg[adj_reg < 50.] = 0
    # G_reg = adjacency_to_graph(adj_reg, reg_labels, 'Time', prune=True)

    # chord_labels, arc_labels = vis._create_node_labels(G_reg, label_col='Time', node_attr='Time')
    # ax = vis.draw_netgraph(G_reg, node_color='b')
    # ax.set_title("Network graph of Regular Segmentation")
    # c_reg = vis.draw_chordgraph(G_reg,
    #                             node_labels=chord_labels,
    #                             label_col='Time',
    #                             title='Chord Graph of Regular Segmentation')
    # ax = vis.draw_arcgraph(G_reg,
    #                        node_size=30.,
    #                        node_labels=arc_labels,
    #                        node_order=range(0, nx.number_of_nodes(G_reg)))
    # ax.set_title("Time-Ordered ArcGraph of Regular Segmentation")
    # vis.show(c_reg)

    # Onset Segmentation
    _, _, segments_onset, adj_onset = mf.thumbnail(audio, fs, length=length, seg_method='onset')

    # Format segment labels
    onset_labels = []
    for i in range(segments_onset.shape[0]):
        onset_labels.append('%2.2f - %2.2f' % (segments_onset[i], segments_onset[i] + length))

    # Create onset segmentation graph
    adj_onset[adj_onset < 50.] = 0
    G_onset = grp.adjacency_matrix_to_graph(adj_onset, onset_labels, label_col, prune=False)
    Gp_onset = grp.prune_graph(G_onset)

    # Test K means clustering (adding 1 to avoid group 0)
    kmeans = cluster_k_means(Gp_onset, k_clusters, n_init=200) + 1
    group_color = kmeans / np.max(kmeans)

    grp.add_node_attribute(Gp_onset, kmeans, node_color_col)
    grp.node_to_edge_attribute(Gp_onset, node_color_col, edge_color_col, from_source=True)

    # Display onset segmentation graph
    chord_labels = grp.to_node_dataframe(Gp_onset)
    arc_labels = grp.to_node_dict(Gp_onset, node_attr=label_col)

    ax = vis.draw_netgraph(Gp_onset, node_color=group_color)
    ax.set_title("Network graph of Onset Segmentation")

    c_onset = vis.draw_chordgraph(Gp_onset,
                                  node_data=chord_labels,
                                  label_col=label_col,
                                  title='Chord Graph Of Onset Segmentation',
                                  node_color=node_color_col,
                                  edge_color=edge_color_col)

    ax = vis.draw_arcgraph(Gp_onset,
                           node_size=30.,
                           node_order=range(0, nx.number_of_nodes(Gp_onset)),
                           node_labels=arc_labels,
                           node_color=group_color
                           )
    ax.set_title("Time-Ordered ArcGraph of Onset Segmentation")

    vis.show(c_onset)
    vis.show()


if __name__ == '__main__':
    main()
