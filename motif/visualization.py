import librosa.display
import numpy as np
import holoviews as hv
import bokeh.plotting as bkplt
import matplotlib.pyplot as plt
import networkx as nx
import arcgraph as arc
import config


def get_axes(ax):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    return ax


# Plot the similarity used to calculate a thumbnail.
def plot_similarity_curve(window_similarity, window_times, labels=None, ax=None):
    ax = get_axes(ax)
    # if ax is None:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 1, 1)

    # Determine axis indices
    # window_start = np.zeros(window_similarity.shape)
    # for i in range(0, window_start.shape[0]):
    #     window_start[i] = i * window_axis

    ax.plot(window_times, window_similarity, 'rx-')
    ax.set_xlabel("Window Starting Point (s)")
    ax.set_ylabel("Window Similarity")
    # ax.title("Self-Similarity Using a Matched Filter")
    if labels is not None:
        ax = add_motif_labels(ax, labels)
    return ax


# Add hand-labeled motifs to a similarity plot for a mf thumbnail
def add_motif_labels(ax, labels):
    ax.set_xlim(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1)
    for i in range(0, labels.shape[0] - 1):
        ax.axvspan(labels.Time[i], labels.Time[i + 1], alpha=0.2, color=labels.Color[i],
                   linestyle='-.', label='Event {}'.format(labels.Event[i]))
    # ax.grid()
    return ax


def plot_similarity_matrix(similarity_matrix, tick_step=3, ax=None):
    ax = get_axes(ax)
    num_windows = similarity_matrix.shape[0]
    # plt.title("Self-Similarity Matrix Using Matched Filter")
    ax.set_xlabel("Window #")
    ax.set_ylabel("Window #")
    ax.xaxis.set_ticks(np.arange(0, num_windows, tick_step))
    ax.yaxis.set_ticks(np.arange(0, num_windows, tick_step))
    image = ax.imshow(similarity_matrix)
    plt.colorbar(image, format='%2.2f')
    return ax


def plot_window_overlap(window_times, window_lengths, tick_step=1, ax=None):
    ax = get_axes(ax)

    num_windows = window_times.shape[0]

    # plt.title("Window layout for {}s windows at {}s intervals".format(window_length, window_step))
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Window #")
    ax.set_ylim(-1, num_windows)
    ax.set_xlim(window_times[0], window_times[num_windows - 1] + window_lengths[num_windows - 1])
    ax.xaxis.set_ticks(np.arange(0, num_windows, tick_step))
    ax.yaxis.set_ticks(np.arange(0, num_windows - 1, tick_step))

    for i in range(0, num_windows):
        line_start = window_times[i]
        line_end = window_times[i] + window_lengths[i]
        ax.axhline(i, line_start, line_end)
        ax.plot([i, line_start], [i, line_end], 'rx')
    return ax


#
# def plot_window_length_comparision(windows_coll, values_coll, lengths, labels=None, ax=None):
#     ax = get_axes(ax)
#     ax.set_xlabel("Snippet Starting Point (s)")
#     ax.set_ylabel("Similarity Score")
#     for i in range(0, len(windows_coll)):
#         ax.plot(windows_coll[i], values_coll[i], 'C{}-'.format(i), label="{} s".format(str(lengths[i])))
#
#     if labels is not None:
#         ax = add_motif_labels(ax, labels)
#     return ax


def plot_stft(sxx, fs=config.FS, ax=None, frames=False,
              hop_length=(config.WINDOW_SIZE * config.OVERLAP_RATIO)):
    ax = get_axes(ax)

    D = librosa.amplitude_to_db(np.abs(sxx), ref=np.max)
    if frames:
        ax = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
                                      hop_length=hop_length, ax=ax)
    else:
        ax = librosa.display.specshow(D, x_axis='time', y_axis='linear',
                                      sr=fs, hop_length=hop_length, ax=ax)
    plt.colorbar(ax.get_children()[0], format='%+2.0f dB', ax=ax)
    return ax


# Plot Spectrogram peaks
def plot_peaks(peaks, ax=None, color='rx'):
    if peaks is None:
        return

    ax = get_axes(ax)
    ax.plot(peaks[:, 1], peaks[:, 0], color)
    ax.set_ylabel('Frequency Frame')
    ax.set_xlabel('Time Frame')
    return ax


# Plot spectogram peaks and one pair from every inc pairs
def plot_pairs(peaks, pairs, inc=50, ax=None,
               line_color='b-',
               peak_color='rx',
               pair_color='k*'):
    if peaks is None or pairs is None:
        return
    ax = get_axes(ax)

    # Choose 1/inc pairs to plot
    pair_mask = np.zeros(pairs.shape[0]).astype(int)
    for i in range(0, pairs.shape[0]):
        if i % inc == 0: pair_mask[i] = i
    pruned = pairs[pair_mask, :]

    ax = plot_peaks(peaks, ax=ax, color=peak_color)

    # Plot connecting lines
    ax.plot([pruned[:, 2], pruned[:, 3]],
            [pruned[:, 0], pruned[:, 1]], line_color)

    # Plot Pairs Starts and Ends
    ax.plot(pruned[:, 2], pruned[:, 0], pair_color)
    ax.plot(pruned[:, 3], pruned[:, 1], pair_color)
    return ax


# Plot everything on the same plot
def plot_stft_with_pairs(sxx, peaks, pairs, inc=50, ax=None,
                         line_color='w-',
                         peak_color='yx',
                         pair_color='c*'):
    if peaks is None or pairs is None:
        return
    ax = get_axes(ax)
    ax = plot_stft(sxx, ax=ax, frames=True)
    ax = plot_pairs(peaks, pairs, inc=inc, ax=ax,
                    line_color=line_color,
                    peak_color=peak_color,
                    pair_color=pair_color)
    return ax


# Draw a simple circular node graph of g
def draw_netgraph(g, ax=None):
    ax = get_axes(ax)
    # ax.title("Graph of Beat Segments w/ circular layout")
    nx.draw(g, pos=nx.circular_layout(g), nodecolor='r', with_labels=True, )
    return ax


# Draw Chord diagram of graph g
def draw_chordgraph(g, size=200):
    data_G = nx.to_pandas_edgelist(g)

    hv.extension('bokeh')
    hv.output(size)
    chord = hv.Chord(data_G)
    chord.opts(
        hv.opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=hv.dim('source').str(),
                      node_color=hv.dim('index').str()))
    c = hv.render(chord, backend='bokeh')
    return c


# Draw arc diagram of graph g
def draw_arcgraph(g, ax=None,
                  node_order=None,
                  node_labels=None,
                  node_color='w',
                  node_size=20.,
                  node_positions=None,
                  edge_width=3.,
                  edge_color='k'):
    adj_G = nx.to_numpy_array(g, weight='value')
    ax = get_axes(ax)

    # Draw arcgraph according to arguments
    if node_positions is not None:
        arc.draw_nodes(node_positions, ax=ax, node_size=node_size, node_color=node_color)
        arc.draw_node_labels(node_positions, node_labels, ax=ax)
        arc.draw_edges(adj_G, ax=ax, node_positions=node_positions, edge_width=edge_width, edge_color=edge_color)
        arc._update_view(adj_G, node_positions, ax=ax)
        arc._make_pretty(ax=ax)

    else:
        arc.draw(adj_G, arc_above=True, ax=ax,
                 node_order=node_order,
                 node_labels=node_labels,
                 node_color=node_color,
                 node_size=node_size,
                 edge_width=edge_width,
                 edge_color=edge_color)
    return ax


def show(chord=None):
    if chord is not None:
        bkplt.show(chord)
    else:
        plt.show()
    return


def main():
    G = nx.generators.sedgewick_maze_graph()
    draw_netgraph(G)
    draw_arcgraph(G)
    c = draw_chordgraph(G, size=500)
    show(c)
    show()
    return


if __name__ == '__main__':
    main()
