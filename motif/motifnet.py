import holoviews as hv
import matplotlib.pyplot as plt
import networkx as nx
import arcgraph as arc


# Draw a simple circular node graph of g
def draw_netgraph(g):
    fig = plt.figure()
    plt.title("Graph of Beat Segments w/ circular layout")
    nx.draw(g, pos=nx.circular_layout(g), nodecolor='r', with_labels=True)
    return fig


# Draw Chord diagram of graph g
def draw_chordgraph(g):
    data_G = nx.to_pandas_edgelist(g)

    hv.extension('bokeh')
    hv.output(size=200)

    chord = hv.Chord(data_G)
    chord.opts(
        hv.opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=hv.dim('source').str(),
                      node_color=hv.dim('index').str()))
    c = hv.render(chord, backend='bokeh')
    return c


# Draw arc diagram of graph g
def draw_arcgraph(g):
    adj_G = nx.to_numpy_array(g, weight='value')

    fig = plt.figure()
    ax = plt.gca()
    arc.draw(adj_G, arc_above=True, ax=ax, node_color='w', node_size=30., edge_width=3., edge_color='k')
    return fig
