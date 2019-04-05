import networkx as nx
import numpy as np
import pandas as pd

import config as cfg


# Clustering and graph analysis functions

def adjacency_matrix_to_graph(adjacency, labels, label_name, prune=False):
    G = nx.DiGraph()
    G = nx.from_numpy_array(adjacency, create_using=G)
    node_idx = np.arange(adjacency.shape[0])

    # For labeling nodes in graphs
    add_node_attribute(G, labels, label_name)

    # For drawing arc graphs, copy the weight attribute to an integer value
    edge_dict = {}
    for u, v in G.edges():
        old_val = G.edges()[u, v]['weight']
        edge_dict[(u, v)] = {'value': int(old_val)}
    nx.set_edge_attributes(G, edge_dict)

    # Prune isolated nodes
    if prune:
        G, node_idx = prune_graph(G)
    return G, node_idx


def graph_to_adjacency_matrix(G, weight_attr='weight'):
    return nx.to_numpy_array(G, weight=weight_attr)


# Generate an incidence matrix directly from an adjacency matrix
# Useful for running clustering more quickly.
def adjacency_to_incidence_matrix(adjacency, prune=False):
    num_nodes = adjacency.shape[0]
    num_edges = np.count_nonzero(adjacency)
    incidence = np.zeros((num_nodes, num_edges))
    label_idx = np.arange(num_nodes)

    (source_list, dest_list) = np.nonzero(adjacency)

    for i in range(num_edges):
        source = source_list[i]
        dest = dest_list[i]
        weight = adjacency[source][dest]
        incidence[source][i] = weight
        incidence[dest][i] = weight

    if prune:
        incidence, relabel_idx = prune_incidence(incidence)
        label_idx = label_idx[relabel_idx]
        # relabels = []
        # mask = np.ones(incidence.shape)
        # num_pruned = 0
        # for i in range(num_nodes):
        #     if np.count_nonzero(incidence[i, :]) == 0:
        #         mask[i, :] = np.zeros(num_edges)
        #         num_pruned += 1
        #     else:
        #         relabels.append(labels[i])
        # incidence = incidence[mask != 0]
        # incidence = incidence.reshape(num_nodes - num_pruned, num_edges)

    return incidence, label_idx


# Construct a graph from a pandas dataframe
def from_pandas_labels(df):
    num_nodes = len(df.index) - 1
    adjacency = np.zeros((num_nodes, num_nodes)).astype('int')
    labels = []
    groups = []
    group_nodes = {}

    # Extract label information
    for i in range(num_nodes):
        seg_label = '{start:2.2f} - {end:2.2f}'.format(start=df.Time[i], end=df.Time[i + 1])
        group_label = df.Event[i]

        if group_label in group_nodes:
            group_nodes[group_label].append(i)
        else:
            group_nodes[group_label] = [i]

        labels.append(seg_label)
        groups.append(group_label)

    # Add edges
    for grp in group_nodes:
        nodes = group_nodes[grp]
        base = nodes[0]
        for i in nodes:
            if i == base:
                continue
            adjacency[i, base] = 50
            # adjacency[base, i] = 1

    G, _ = adjacency_matrix_to_graph(adjacency, labels, cfg.NODE_LABEL, prune=False)
    add_node_attribute(G, groups, cfg.CLUSTER_NAME)
    node_to_edge_attribute(G, cfg.CLUSTER_NAME, cfg.CLUSTER_EDGE_NAME, from_source=True)

    return G


# Returns a shallow copy of graph g with no isolated nodes.
def prune_graph(g):
    D = nx.DiGraph(g)
    isolate_idx = list(nx.isolates(D))
    D.remove_nodes_from(isolate_idx)
    D = nx.convert_node_labels_to_integers(D)

    label_idx = np.arange(len(g.nodes))
    mask = np.ones(label_idx.shape, dtype=int)
    isolate_idx = np.array(isolate_idx, dtype=int)
    mask[isolate_idx] = 0
    label_idx = label_idx[mask == 1]
    return D, label_idx


def prune_incidence(incidence):
    label_idx = []
    num_nodes = incidence.shape[0]
    num_edges = incidence.shape[1]
    mask = np.ones(incidence.shape)
    num_pruned = 0
    for i in range(num_nodes):
        if np.count_nonzero(incidence[i, :]) == 0:
            mask[i, :] = np.zeros(num_edges)
            num_pruned += 1
        else:
            label_idx.append(i)
    incidence = incidence[mask != 0]
    incidence = incidence.reshape(num_nodes - num_pruned, num_edges)

    return incidence, label_idx


# Add a node attribute from an array.
def add_node_attribute(g, node_attribute, attr_name):
    for i in range(nx.number_of_nodes(g)):
        nx.set_node_attributes(g, {i: {attr_name: node_attribute[i]}})


# Add an edge attribute  from a 2d array
def add_edge_attribute(g, edge_attribute, attr_name):
    for u, v in g.edges():
        nx.set_edge_attributes(g, {(u, v): {attr_name: edge_attribute[u, v]}})


# Add an attribute from a node to each edge
def node_to_edge_attribute(g, node_attr, edge_attr, from_source=True):
    for u, v in g.edges():
        if from_source:
            node_val = g.nodes()[u][node_attr]
        else:
            node_val = g.nodes()[v][node_attr]
        nx.set_edge_attributes(g, {(u, v): {edge_attr: node_val}})


# Create node dataset for a chord graph using all node attributes.
def to_node_dataframe(g):
    node_data = {}
    for i in g.nodes():
        for attr in g.nodes()[i]:
            if attr in node_data:
                node_data[attr].append(g.nodes()[i][attr])
            else:
                node_data[attr] = [g.nodes()[i][attr]]

    return pd.DataFrame(node_data)


# Creates a dictionary of node indices to labels using the node attribute as a label.
def to_node_dict(g, node_attr='label'):
    node_dict = {}
    for i in g.nodes():
        node_dict[i] = g.nodes()[i][node_attr]
    return node_dict


# Condense to a minor graph, using merge_attr as the new graph indices.
# Maintains only the sum of weight_attr of edges between different groups.
def condense_by_attr(g, merge_attr, weight_attr='weight'):
    D = nx.DiGraph()

    # Construct the merged graph
    node_data = {}
    for i in g.nodes():
        group = g.nodes()[i][merge_attr]

        # Add new node if not in graph
        if group not in D:
            D.add_node(group)

        # Add node attributes to graph
        for attr in g.nodes()[i]:
            if attr is merge_attr:
                continue
            attr_data = g.nodes()[i][attr]

            if group in node_data:
                if attr in node_data[group]:
                    node_data[group][attr].append(attr_data)
                else:
                    node_data[group][attr] = [attr_data]
            else:
                node_data[group] = {attr: [attr_data]}

    nx.set_node_attributes(D, node_data)

    # Add and sum edge weights
    edge_data = {}
    for u, v in g.edges():
        u_d = g.nodes()[u][merge_attr]
        v_d = g.nodes()[v][merge_attr]

        # No self loops
        if u_d == v_d:
            continue

        if (u_d, v_d) not in D.edges():
            D.add_edge(u_d, v_d)

        attr_data = g.edges()[u, v][weight_attr]
        if (u_d, v_d) in edge_data:
            edge_data[(u_d, v_d)] += attr_data
        else:
            edge_data[(u_d, v_d)] = attr_data

    nx.set_edge_attributes(D, edge_data, weight_attr)

    return D


def main():
    # Simple adjacency matrix
    adj = np.array([[0.1, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    # adj_2 = np.copy(adj)
    # adj_2[adj_2 < 1.] = 0
    labels = ['Batman', 'Bane', 'Joker', 'Robin']
    # label_col= 'Super Hero'
    #
    # G = adjacency_matrix_to_graph(adj, labels, label_col, prune=False)
    # G_thresh = adjacency_matrix_to_graph(adj_2, labels, label_col, prune=False)
    # G_prune = prune_graph(G)
    # print("Regular Graph: \n Nodes: {} \n Edges: {}".format(G.nodes(), G.edges()))
    # print("Thresholded Graph: \n Nodes: {} \n Edges: {}".format(G_thresh.nodes(), G_thresh.edges()))
    # print("Pruned Graph: \n Nodes: {} \n Edges: {}".format(G_prune.nodes(), G_prune.edges()))
    print(adjacency_to_incidence_matrix(adj, prune=False))
    print(adjacency_to_incidence_matrix(adj, prune=True))

    # name = 't3_train'
    # directory = "./bin/labelled"
    # audio_labels = fileutils.load_labels(name, label_dir=directory)
    # G = from_pandas_labels(audio_labels)
    # print(G.nodes[0])
    # print(type(G))


if __name__ == '__main__':
    main()
