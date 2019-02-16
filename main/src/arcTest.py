import numpy as np
import matplotlib.pyplot as plt
import network_line_graph as nlg

# initialise figure
fig, ax = plt.subplots(1, 1)

# make a weighted random graph
n = 20  # number of nodes
p = 0.1  # connection probability
a1 = np.random.choice([False, True], size=(n, n), replace=True, p=[1-p, p])
# w1 = a1
w1 = np.random.randn(n, n)  # weight matrix
w1[~a1] = 0
order = np.arange(0, n)
# print(order.shape)
print(np.max(w1), np.min(w1))
#
# # plot connections above x-axis
nlg.draw(w1, arc_above=True, ax=ax)
#
# make another weighted random graph;
a2 = np.random.rand(n, n) < p  # adjacency matrix
w2 = np.random.randn(n, n)  # weight matrix
w2[~a2] = 0

# plot connections below x-axis
nlg.draw(w2, arc_above=False, ax=ax)

# annotate
ax.text(0, 1, 'Graph 1', transform=ax.transAxes, fontsize=18)
ax.text(0, 0, 'Graph 2', transform=ax.transAxes, fontsize=18)

plt.show()
