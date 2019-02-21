import networkx as nx

import fileutils
import visualization as vis
import matchfilt as mf
import visualization as vis


# Clustering and graph analysis functions
def add_node_attribute(g, node_attribute, attr_name):
    for i in range(nx.number_of_nodes(g)):
        nx.set_node_attributes(g, {i: {attr_name: node_attribute[i]}})


def add_edge_attribute(g, edge_attribute, attr_name):
    for u, v in g.edges():
        nx.set_edge_attributes(g, {(u, v): {attr_name: edge_attribute[u, v]}})


def copy_edge_attribute(g, old_attr, new_attr):
    for u, v in g.edges():
        old_val = g.edges[u, v][old_attr]
        nx.set_edge_attributes(g, {(u, v): {new_attr: old_val}})


def main():
    name = 't3_train'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)

    # Regular Segmentation
    G_reg = nx.DiGraph()
    _, _, segments_reg, adj_reg = mf.thumbnail(audio, fs, length=2)
    adj_reg[adj_reg < 50.] = 0
    G_reg = nx.from_numpy_array(adj_reg, create_using=G_reg)
    add_node_attribute(G_reg, segments_reg, 'Start Time')
    copy_edge_attribute(G_reg, old_attr='weight', new_attr='value')

    chord_labels, arc_labels = vis._create_node_labels(G_reg, node_attr='Start Time')
    ax = vis.draw_netgraph(G_reg)
    ax.set_title("Network graph of Regular Segmentation")
    c_reg = vis.draw_chordgraph(G_reg,
                                node_labels=chord_labels,
                                label_col='label',
                                title='Chord Graph of Regular Segmentation')
    ax = vis.draw_arcgraph(G_reg, node_size=50., node_order=range(0, nx.number_of_nodes(G_reg)))
    ax.set_title("Time-Ordered ArcGraph of Regular Segmentation")

    # Onset Segmentation
    G_onset = nx.DiGraph()
    _, _, segments_onset, adj_onset = mf.thumbnail(audio, fs, length=2, seg_method='onset')
    adj_onset[adj_onset < 50.] = 0
    G_onset = nx.from_numpy_array(adj_onset, create_using=G_onset)
    add_node_attribute(G_onset, segments_onset, 'Start Time')
    copy_edge_attribute(G_onset, old_attr='weight', new_attr='value')

    chord_labels, arc_labels = vis._create_node_labels(G_onset, node_attr='Start Time')
    ax = vis.draw_netgraph(G_onset)
    ax.set_title("Network graph of Onset Segmentation")
    c_onset = vis.draw_chordgraph(G_onset,
                                  node_labels=chord_labels,
                                  label_col='label',
                                  title='Chord Graph Of Onset Segmentation')
    ax = vis.draw_arcgraph(G_onset, node_size=50., node_order=range(0, nx.number_of_nodes(G_onset)))
    ax.set_title("Time-Ordered ArcGraph of Onset Segmentation")

    vis.show(c_onset)
    vis.show(c_reg)
    vis.show()


if __name__ == '__main__':
    main()


