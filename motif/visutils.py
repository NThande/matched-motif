import librosa.display
import numpy as np
import holoviews as hv
import bokeh.plotting as bkplt
import matplotlib.pyplot as plt
import networkx as nx
import arcgraph as arc
import pandas as pd
import config as cfg
import graphutils as grp


def get_axes(ax):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    return ax


# Plot the similarity used to calculate a thumbnail.
def plot_similarity_curve(seg_similarity, segment_times, labels=None, ax=None, color='rx-'):
    ax = get_axes(ax)
    ax.plot(segment_times, seg_similarity, color)
    ax.set_xlabel("Window Starting Point (s)")
    ax.set_ylabel("Window Similarity")
    if labels is not None:
        ax = add_motif_labels(ax, labels)
    return ax


# Add hand-labeled motifs to a similarity plot for a mf thumbnail
def add_motif_labels(ax, labels):
    ax.set_xlim(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1)
    for i in range(0, labels.shape[0] - 1):
        ax.axvspan(labels.Time[i], labels.Time[i + 1], alpha=0.2, color=labels.Color[i],
                   linestyle='-.', label='Event {}'.format(labels.Event[i]))
    return ax


def plot_similarity_matrix(similarity_matrix, tick_step=3, ax=None):
    ax = get_axes(ax)
    num_windows = similarity_matrix.shape[0]
    ax.set_xlabel("Window #")
    ax.set_ylabel("Window #")
    ax.xaxis.set_ticks(np.arange(0, num_windows, tick_step))
    ax.yaxis.set_ticks(np.arange(0, num_windows, tick_step))
    image = ax.imshow(similarity_matrix)
    cbar = plt.colorbar(image, format='%2.0f')
    cbar.set_label('Similarity')
    return ax


def plot_window_overlap(segments, seg_lengths, tick_step=1, ax=None):
    ax = get_axes(ax)

    num_windows = segments.shape[0]
    min_seg = segments[0]
    max_seg = segments[num_windows - 1] + seg_lengths[num_windows - 1]

    # plt.title("Window layout for {}s windows at {}s intervals".format(window_length, window_step))
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Window #")
    ax.set_ylim(-1, num_windows)
    ax.set_xlim(min_seg, max_seg)
    ax.xaxis.set_ticks(np.arange(0, int(max_seg), tick_step))
    ax.yaxis.set_ticks(np.arange(0, num_windows - 1, tick_step))

    for i in range(0, num_windows):
        line_start = segments[i]
        line_end = segments[i] + seg_lengths[i]
        ax.axhline(i, line_start / max_seg, line_end / max_seg)
        ax.plot([line_start, line_end], [i, i], 'rx')
    return ax


def plot_stft(sxx, fs=cfg.FS, ax=None, frames=False,
              hop_length=(cfg.WINDOW_SIZE * cfg.OVERLAP_RATIO)):
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
    idx_f = cfg.FREQ_IDX
    ax.plot(peaks[:, idx_f], peaks[:, idx_f], color)
    ax.set_ylabel('Frequency Frame')
    ax.set_xlabel('Time Frame')
    return ax


# Plot spectrogram peaks and one pair from every inc pairs
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

    idx_f = cfg.FREQ_IDX
    idx_t = cfg.PAIR_TIME_IDX
    # Plot connecting lines
    ax.plot([pruned[:, idx_t], pruned[:, idx_t + 1]],
            [pruned[:, idx_f], pruned[:, idx_f + 1]], line_color)

    # Plot Pairs Starts and Ends
    ax.plot(pruned[:, idx_t], pruned[:, idx_f], pair_color)
    ax.plot(pruned[:, idx_t + 1], pruned[:, idx_f + 1], pair_color)
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


# Draw Chord diagram of graph g
def draw_chordgraph(g,
                    node_data=None,
                    label_col='index',
                    size=200,
                    cmap='Category20',
                    title='',
                    draw_labels=True,
                    edge_color='source',
                    node_color='index'):
    hv.extension('bokeh')
    hv.output(size=size)

    # Get the edge list of the graph
    edge_g = nx.to_pandas_edgelist(g)

    # Account for passed in node dataset
    if node_data is not None:
        node_dataset = hv.Dataset(pd.DataFrame(node_data), 'index')
        chord = hv.Chord((edge_g, node_dataset), label=title).select(value=(5, None))
    else:
        chord = hv.Chord(edge_g, label=title)
        label_col = 'index'

    # Draw the desired graph
    if draw_labels is True:
        chord.opts(
            hv.opts.Chord(
                cmap=cmap, edge_cmap=cmap,
                edge_color=hv.dim(edge_color).str(),
                node_color=hv.dim(node_color).str(),
                labels=label_col
            ))
    else:
        chord.opts(
            hv.opts.Chord(cmap=cmap, edge_cmap=cmap,
                          edge_color=hv.dim(edge_color).str(),
                          node_color=hv.dim(node_color).str()
                          ))
    c = hv.render(chord, backend='bokeh')
    return c


# Draw a simple circular node graph of g
def draw_netgraph(g, ax=None, **kwargs):
    ax = get_axes(ax)
    node_layout = kwargs.setdefault('node_layout', nx.circular_layout(g))
    nx.draw(g, pos=node_layout,
            with_labels=True,
            arrows=True,
            **kwargs)
    return ax


# Draw arc diagram of graph g
def draw_arcgraph(g, ax=None,
                  weight_attr='value',
                  node_order=None,
                  node_labels=None,
                  node_color='w',
                  node_size=20.,
                  node_positions=None,
                  edge_width=3.,
                  edge_color=None,
                  **kwargs
                  ):
    adj_G = nx.to_numpy_array(g, weight=weight_attr)
    vertical_shift = kwargs.setdefault('vertical_shift', 2)
    ax = get_axes(ax)

    # Draw arcgraph according to arguments
    if node_positions is not None:
        arc.draw_nodes(node_positions, ax=ax,
                       node_size=node_size,
                       node_color=node_color)
        arc.draw_node_labels(node_positions, node_labels, ax=ax,
                             vertical_shift=vertical_shift)
        arc.draw_edges(adj_G, ax=ax,
                       node_positions=node_positions,
                       edge_width=edge_width,
                       edge_color=edge_color)
        arc._update_view(adj_G, node_positions, ax=ax)
        arc._make_pretty(ax=ax)

    else:
        arc.draw(adj_G, arc_above=True, ax=ax,
                 node_order=node_order,
                 node_labels=node_labels,
                 node_color=node_color,
                 node_size=node_size,
                 edge_width=edge_width,
                 vertical_shift=vertical_shift
                 )
    return ax


def show(chord=None):
    if chord is not None:
        bkplt.show(chord)
    else:
        plt.show()
    return


def main():
    G = nx.generators.sedgewick_maze_graph()
    for i in range(nx.number_of_nodes(G)):
        nx.set_node_attributes(G, {i: {'label': 'N{}'.format(i)}})
    for u, v in G.edges():
        nx.set_edge_attributes(G, {(u, v): {'value': np.random.randint(0, 100)}})
    node_data = grp.to_node_dataframe(G)
    node_dict = grp.to_node_dict(G)
    draw_netgraph(G)
    draw_arcgraph(G, node_labels=node_dict, node_size=35.)
    c = draw_chordgraph(G, size=200, node_data=node_data, label_col='label')
    show(c)
    show()
    return


if __name__ == '__main__':
    main()
