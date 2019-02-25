import networkx as nx
import pandas as pd
import numpy as np


# Clustering and graph analysis functions
def adjacency_matrix_to_graph(adj, labels, label_name, prune=False):
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


# Add an attribute from a node to each edge
def node_to_edge_attribute(g, node_attr, edge_attr, from_source=True):
    for u, v in g.edges:
        if from_source:
            node_val = g.nodes[u][node_attr]
        else:
            node_val = g.nodes[v][node_attr]
        nx.set_edge_attributes(g, {(u, v): {edge_attr: node_val}})


# Create node dataset for a chord graph using all node attributes.
def to_node_dataframe(g):
    node_data = {}
    for i in g.nodes:
        for attr in g.nodes[i]:
            if attr in node_data:
                node_data[attr].append(g.nodes[i][attr])
            else:
                node_data[attr] = [g.nodes[i][attr]]

    return pd.DataFrame(node_data)


# Creates a dictionary of node indices to labels using the node attribute as a label.
def to_node_dict(g, node_attr='label'):
    node_dict = {}
    for i in g.nodes:
        node_dict[i] = g.nodes[i][node_attr]
    return node_dict


def main():
    # Simple adjacency matrix
    adj = np.array([[0.1, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    adj_2 = np.copy(adj)
    adj_2[adj_2 < 1.] = 0
    labels = ['Batman', 'Bane', 'Joker', 'Robin']
    label_col= 'Super Hero'

    G = adjacency_matrix_to_graph(adj, labels, label_col, prune=False)
    G_thresh = adjacency_matrix_to_graph(adj_2, labels, label_col, prune=False)
    G_prune = prune_graph(G)
    print("Regular Graph: \n Nodes: {} \n Edges: {}".format(G.nodes, G.edges))
    print("Thresholded Graph: \n Nodes: {} \n Edges: {}".format(G_thresh.nodes, G_thresh.edges))
    print("Pruned Graph: \n Nodes: {} \n Edges: {}".format(G_prune.nodes, G_prune.edges))


if __name__ == '__main__':
    main()
