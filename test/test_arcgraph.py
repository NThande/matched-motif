import numpy as np
import matplotlib.pyplot as plt
import arcgraph as arc


def test_arcgraph():
    # initialise figure
    fig, ax = plt.subplots(1, 1)

    # make a weighted random graph
    n = 20  # number of nodes
    p = 0.05  # connection probability
    a1 = np.random.rand(n, n) < p
    w1 = np.random.randn(n, n)  # weight matrix
    w1[~a1] = 0

    # plot connections above x-axis
    arc.draw(w1, arc_above=True, ax=ax)

    plt.show()
