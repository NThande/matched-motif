import networkx as nx
import fileutils
import matchfilt as mf
import numpy as np
from sklearn import cluster
import visutils as vis
import graphutils as grp


# Cluster using k means with k_clusters clusters
def cluster_k_means(g, k_clusters, incidence=None, weight='weight', n_init=200):
    if incidence is None:
        incidence = nx.incidence_matrix(g, weight=weight)
    kmeans_clf = cluster.KMeans(n_clusters=k_clusters, n_init=n_init)
    kmeans = kmeans_clf.fit_predict(incidence)
    return kmeans


# Cluster using agglomerative clustering, starting with k_clusters clusters.
def cluster_agglom(g, k_clusters=2, incidence=None, weight='weight', linkage='ward'):
    if incidence is None:
        incidence = nx.incidence_matrix(g, weight=weight)
    agglom_clf = cluster.AgglomerativeClustering(n_clusters=k_clusters, linkage=linkage)
    agglom = agglom_clf.fit_predict(incidence)
    return agglom


# Condense to a minor graph, using merge_attr as the new graph indices. Maintains only the weight_attr between edges.
def condense(g, merge_attr, weight_attr='weight'):
    D = nx.DiGraph()

    # Construct the merged graph
    node_data = {}
    for i in g.nodes:
        group = g.nodes[i][merge_attr]

        # Add new node if not in graph
        if group not in D:
            D.add_node(group)

        # Add node attributes to graph
        for attr in g.nodes[i]:
            if attr is merge_attr:
                continue
            attr_data = g.nodes[i][attr]

            if group in node_data:
                if attr in node_data[group]:
                    node_data[group][attr].append(attr_data)
                else:
                    node_data[group][attr] = [attr_data]
            else:
                node_data[group] = {attr: [attr_data]}

    nx.set_node_attributes(D, node_data)

    edge_data = {}
    for u, v in g.edges:
        u_d = g.nodes[u][merge_attr]
        v_d = g.nodes[v][merge_attr]

        for attr in g.edges[u, v]:
            attr_data = g.edges[u, v][attr]
            if (u_d, v_d) in edge_data:
                if attr in edge_data[u_d, v_d]:
                    edge_data[u_d, v_d][attr].append(attr_data)
                else:
                    edge_data[u_d, v_d][attr] = [attr_data]
            else:
                edge_data[u_d, v_d] = {attr: [attr_data]}

    print(nx.to_pandas_edgelist(D))
    return D
    # print(node_data)


def main():
    name = 't3_train'
    directory = "./bin/labelled"
    audio, fs = fileutils.load_audio(name, audio_dir=directory)
    length = 3
    k_clusters = 3
    label_name = 'Time'
    cluster_node_name = 'Group'
    cluster_edge_name = 'Edge Group'

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

    # chord_labels, arc_labels = vis._create_node_labels(G_reg, label_name='Time', node_attr='Time')
    # ax = vis.draw_netgraph(G_reg, node_color='b')
    # ax.set_title("Network graph of Regular Segmentation")
    # c_reg = vis.draw_chordgraph(G_reg,
    #                             node_labels=chord_labels,
    #                             label_name='Time',
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
    G_onset = grp.adjacency_matrix_to_graph(adj_onset, onset_labels, label_name, prune=False)
    Gp_onset = grp.prune_graph(G_onset)

    # Test K means clustering (adding 1 to avoid group 0)
    kmeans = cluster_k_means(Gp_onset, k_clusters, n_init=200)
    group_color = kmeans / np.max(kmeans)

    grp.add_node_attribute(Gp_onset, kmeans, cluster_node_name)
    grp.node_to_edge_attribute(Gp_onset, cluster_node_name, cluster_edge_name, from_source=True)

    condense(Gp_onset, cluster_node_name)

    # # Display onset segmentation graph
    # chord_labels = grp.to_node_dataframe(Gp_onset)
    # arc_labels = grp.to_node_dict(Gp_onset, node_attr=label_name)
    #
    # c_onset = vis.draw_chordgraph(Gp_onset,
    #                               node_data=chord_labels,
    #                               label_col=label_name,
    #                               title='Chord Graph Of Onset Segmentation')
    #
    # c_onset_k = vis.draw_chordgraph(Gp_onset,
    #                                 node_data=chord_labels,
    #                                 label_col=label_name,
    #                                 title='Chord Graph Of Onset Segmentation '
    #                                       'with {}-means clustering'.format(k_clusters),
    #                                 node_color=cluster_node_name,
    #                                 edge_color=cluster_edge_name)
    #
    # ax = vis.draw_netgraph(Gp_onset, node_color=group_color)
    # ax.set_title("Network graph of Onset Segmentation")
    #
    # ax = vis.draw_arcgraph(Gp_onset,
    #                        node_size=30.,
    #                        node_order=range(0, nx.number_of_nodes(Gp_onset)),
    #                        node_labels=arc_labels,
    #                        node_color=group_color
    #                        )
    # ax.set_title("Time-Ordered ArcGraph of Onset Segmentation")
    #
    # vis.show(c_onset)
    # vis.show(c_onset_k)
    # vis.show()


if __name__ == '__main__':
    main()
