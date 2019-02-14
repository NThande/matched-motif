from pathlib import Path

import holoviews as hv
import librosa as lb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from bokeh.plotting import show

hv.extension('bokeh')

# Read in the audio
audio_dir = Path("C:/Users/nthan/PycharmProjects/autosampler/main/bin/")
file_name = audio_dir / "t1.wav"
audio, fs = lb.load(file_name)

# Identify beats in the audio
tempo, beats = lb.beat.beat_track(y=audio, sr=fs, units='samples')
print("Number of beat indices: ", beats.shape[0])
beats = np.concatenate([[0], beats, [audio.shape[0]]])

# Construct graph from beats
G = nx.DiGraph(tempo=tempo)
if tempo != 0 :
    for i in range(0, beats.shape[0] - 1):
        G.add_node(i, start=beats[i], end=beats[i+1], length=(beats[i+1] - beats[i]))
print("Number of graph nodes: ", G.number_of_nodes())
# print(G.nodes.data())

# Calculate weighted edges
min_diff = 1.0 * fs
for i in range(0, G.number_of_nodes()):
    for j in range(0, G.number_of_nodes()):
        if i == j: continue
        e_weight = np.abs(G.node[i]['length'] - G.node[j]['length'])
        if e_weight == 0:
            # e_weight_adj = e_weight / fs
            e_weight_adj = 1.0/(j + 1)
            G.add_edge(i, j, value= e_weight_adj, weight=e_weight_adj)

print("Number of Graph Edges: ", G.number_of_edges())

# Draw the graph
plt.figure()
plt.title("Graph of Beat Segments w/ circular layout")
nx.draw(G, pos=nx.circular_layout(G), nodecolor='r', with_labels=True)

print("Groups of Nodes with same length: ", nx.number_strongly_connected_components(G))
scc_G = nx.strongly_connected_components(G)
for c in scc_G:
    print(c)
    for i in c:
        print(G.node[i]['start'])
data_G = nx.to_pandas_edgelist(G)
# print(data_G)

hv.output(size=200)

print(data_G.head(3))

c = hv.render(hv.Chord(data_G), backend='bokeh')
print(type(c))
show(c)

plt.show()