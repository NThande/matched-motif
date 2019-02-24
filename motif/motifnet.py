import networkx as nx

import fileutils
import matchfilt as mf
import visualization as vis


# Clustering and graph analysis functions
def adjacency_to_graph(adj, labels, label_name, prune=False):
    G = nx.DiGraph()
    G = nx.from_numpy_array(adj, create_using=G)

    # For labeling nodes in graphs
    add_node_attribute(G, labels, label_name)

    # For drawing arc graphs
    copy_edge_attribute(G, old_attr='weight', new_attr='value')

    # Prune isolated nodes
    if prune:
        G = prune_graph(G)
    return G


# Returns a shallow copy of graph g with no isolated nodes.
def prune_graph(g):
    D = nx.DiGraph(g)
    D.remove_nodes_from(list(nx.isolates(D)))
    D = nx.convert_node_labels_to_integers(D)
    return D


# Add a node attribute from an array.
def add_node_attribute(g, node_attribute, attr_name):
    for i in range(nx.number_of_nodes(g)):
        nx.set_node_attributes(g, {i: {attr_name: node_attribute[i]}})


# Add an edge attribute  from a 2d array
def add_edge_attribute(g, edge_attribute, attr_name):
    for u, v in g.edges():
        nx.set_edge_attributes(g, {(u, v): {attr_name: edge_attribute[u, v]}})


# Copy a named edge attribute in graph g
def copy_edge_attribute(g, old_attr, new_attr):
    for u, v in g.edges():
        old_val = g.edges[u, v][old_attr]
        nx.set_edge_attributes(g, {(u, v): {new_attr: old_val}})


def clustering(g):
    return nx.clustering(g)


def main():
    name = 't3_train'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    length = 3

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
    G_onset = adjacency_to_graph(adj_onset, onset_labels, 'Time', prune=False)
    Gp_onset = prune_graph(G_onset)


    # Display onset segmentation graph
    # chord_labels, arc_labels = vis._create_node_labels(Gp_onset, label_col='Time', node_attr='Time')
    # ax = vis.draw_netgraph(G_onset, node_color='b')
    # ax.set_title("Network graph of Onset Segmentation")
    # c_onset = vis.draw_chordgraph(Gp_onset,
    #                               node_labels=chord_labels,
    #                               label_col='Time',
    #                               title='Chord Graph Of Onset Segmentation')
    # ax = vis.draw_arcgraph(Gp_onset,
    #                        node_size=30.,
    #                        node_order=range(0, nx.number_of_nodes(Gp_onset)),
    #                        node_labels=arc_labels
    #                        )
    # ax.set_title("Time-Ordered ArcGraph of Onset Segmentation")
    # vis.show(c_onset)

    # vis.show()


if __name__ == '__main__':
    main()
