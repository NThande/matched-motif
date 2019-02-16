import numpy as np
import matplotlib.pyplot as plt
import arcgraph as arc

# initialise figure
fig, ax = plt.subplots(1, 1)

# make a weighted random graph
n = 20  # number of nodes
p = 0.05  # connection probability
a1 = np.random.rand(n, n) < p
w1 = np.random.randn(n, n)  # weight matrix
w1[~a1] = 0
print(np.max(w1), np.min(w1))

# plot connections above x-axis
arc.draw(w1, arc_above=True, ax=ax)

# make another weighted random graph;
a2 = np.random.rand(n, n) < p  # adjacency matrix
w2 = np.random.randn(n, n)  # weight matrix
w2[~a2] = 0

# plot connections below x-axis
arc.draw(w2, arc_above=False, ax=ax)

# annotate
ax.text(0, 1, 'Graph 1', transform=ax.transAxes, fontsize=18)
ax.text(0, 0, 'Graph 2', transform=ax.transAxes, fontsize=18)

plt.show()
