import bokeh.plotting as bkplt
import holoviews as hv
import librosa.display
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib as mpl
import arcgraph as arc
import config as cfg
import graphutils as graph
import motifutils as motif

mpl.style.use('seaborn-white')
mpl.rcParams.update({'font.size': 16,
                     'axes.titlesize': 22,
                     'axes.labelsize': 20,
                     'figure.titlesize': 24,
                     'figure.titleweight': 'bold'})
plt.rcParams['image.cmap'] = 'plasma'


# Gives you a figure handle
def get_fig():
    fig = plt.figure()
    fig.set_size_inches(8.0, 5.5)
    return fig


# Saves the figure handle fig to directory out_dir with the given name and dpi
def save_fig(fig, out_dir, name, dpi=150):
    fig.savefig(out_dir + name + ".png", dpi=dpi)


# Gets you an axes handle from a new figure, if one does not already exist.
def get_axes(ax):
    if ax is None:
        fig = get_fig()
        ax = fig.add_subplot(1, 1, 1)
    return ax


# Plot the similarity used to calculate a thumbnail.
def plot_similarity_curve(seg_similarity, segment_times, labels=None, ax=None, color='rx-'):
    ax = get_axes(ax)
    ax.plot(segment_times, seg_similarity, color)
    ax.set_xlabel("Window Starting Point (s)")
    ax.set_ylabel("Window Similarity")
    if labels is not None:
        ax = add_motif_labels_with_df(ax, labels)
    return ax


# Add hand-labeled motifs to a similarity plot for a matched filter thumbnail
def add_motif_labels_with_df(ax, labels_df, alpha=0.2):
    starts, ends, labels = motif.df_to_motif(labels_df)
    ax = add_motif_labels(ax, starts, ends, labels, alpha=alpha)
    return ax


# Add translucent colored boxes to represent motifs to a plot. ax x-axis must be in seconds for this to work properly.
def add_motif_labels(ax, starts, ends, labels, alpha=0.8):
    ax.set_xlim(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1)
    unique_motifs = {}
    linewidth = 3
    for i in range(0, starts.shape[0]):
        this_label = int(labels[i])
        if this_label in unique_motifs:
            ax.axvspan(starts[i], ends[i], alpha=alpha, facecolor='C{}'.format(labels[i] % 10),
                       linestyle='-', edgecolor='k', linewidth=linewidth)
        else:
            unique_motifs[this_label] = 1
            ax.axvspan(starts[i], ends[i], alpha=alpha, facecolor='C{}'.format(this_label % 10),
                       linestyle='-', edgecolor='k', linewidth=linewidth, label='Motif {}'.format(this_label))
    return ax


# Plot a self-similarity matrix on its own plot
def plot_similarity_matrix(similarity_matrix, tick_step=3, ax=None):
    ax = get_axes(ax)
    ax, image = plot_matrix(similarity_matrix, ax)
    num_windows = similarity_matrix.shape[0]
    ax.set_xlabel("Window #")
    ax.set_ylabel("Window #")
    ax.xaxis.set_ticks(np.arange(0, num_windows, tick_step))
    ax.yaxis.set_ticks(np.arange(0, num_windows, tick_step))
    cbar = plt.colorbar(image, format='%2.2f')
    cbar.set_label('Similarity')
    return ax


# Plot a generic matrix on the given axis, with matrix indices increasing from bottom left to top right corner
def plot_matrix(matrix, ax=None):
    ax = get_axes(ax)
    image = ax.imshow(matrix)
    ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
    return ax, image


# Plot the overlap of the segments in a staircase format
def plot_window_overlap(segments, seg_lengths, audio_len, tick_step=1, ax=None):
    ax = get_axes(ax)

    num_windows = segments.shape[0]
    min_seg = 0
    max_seg = audio_len

    ax.set_ylim(-1, num_windows + 1)
    ax.set_xlim(min_seg, max_seg)
    audio_step = int(audio_len/4)
    ax.xaxis.set_ticks(np.arange(0, audio_len + audio_step, audio_step))
    ax.yaxis.set_ticks(np.arange(0, num_windows + tick_step, tick_step))

    for i in range(0, num_windows):
        line_start = segments[i]
        line_end = segments[i] + seg_lengths[i]
        ax.axhspan(i, i+1, line_start / max_seg, line_end / max_seg, linewidth=2, edgecolor='k', facecolor='w')
    return ax


# Plot the input matrix sxx as a Fourier spectrogram. Doesn't work if it's not a Foureier spectrogram.
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


# Plot peak coordinates as points on the given ax.
def plot_peaks(peaks, ax=None, color='rx'):
    if peaks is None:
        return

    ax = get_axes(ax)
    idx_f = cfg.FREQ_IDX
    ax.plot(peaks[:, idx_f + 1], peaks[:, idx_f], color)
    ax.set_ylabel('Frequency Frame')
    ax.set_xlabel('Time Frame')
    return ax


# Plot all spectrogram peaks on the given ax. Then, plot 1/inc peak pairs as lines between peaks.
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
        if i % inc == 0:
            pair_mask[i] = i
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


# Plot the spectrogram overlaid with peak locations and pairs.
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


# Plot the waveform of the audio and the motif segments on the same plot.
def plot_motif_segmentation(audio, fs, starts, ends, labels, ax=None, alpha=0.5):
    ax = get_axes(ax)
    plot_audio_waveform(audio, fs, ax)
    labels = labels.astype(int)
    add_motif_labels(ax, starts, ends, labels, alpha)
    audio_len = int(audio.shape[0] / fs)
    plt.xticks(np.arange(audio_len + 1, step=np.ceil(audio_len / 10)))
    ax.legend(frameon=True).set_draggable(state=True)
    return ax


# Plot an audio waveform. Doesn't work if it's not an audio waveform.
def plot_audio_waveform(audio, fs, ax=None):
    ax = get_axes(ax)
    librosa.display.waveplot(audio, fs, ax=ax, color='gray')
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Amplitude")
    return ax


# Plots a bar graph on the given axis with the given parameters.
# metric_labels are the x-tick labels, group_label is the legend label for this plot.
def plot_metric_bar(x_pos, values, ax=None,
                    metric_labels=None, color='b', group_label=None, width=0.8):
    ax = get_axes(ax)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.bar(x_pos, values,
           width=width,
           tick_label=metric_labels,
           linewidth=3,
           edgecolor='black',
           color=color,
           label=group_label)
    return ax


# Plots a generic curve; group_label is the legend label.
def plot_metric_curve(x_pos, values, group_label=None, ax=None, color='rx-'):
    ax = get_axes(ax)
    ax.plot(x_pos, values, color, label=group_label)
    return ax


# Draw a chord diagram of the graph G. Handy way to visualize some self-similarity matrices.
# node_data is the dataframe for the node labels; label_col is the column of the node_data used for node labels.
# node_data should be made with the graphutils function to_node_dataframe().
def plot_chordgraph(G,
                    node_data=None,
                    label_col='index',
                    size=200,
                    cmap='Category20',
                    title='',
                    draw_labels=True,
                    edge_color='E-Group',
                    node_color='index'):
    hv.extension('bokeh')
    hv.output(size=size)

    # Get the edge list of the graph
    H = nx.to_undirected(G)
    edge_data = nx.to_pandas_edgelist(H)

    # Enforce that the value column is in the right position in the dataframe
    print(edge_data.columns)
    val_loc = 2
    cur_loc = edge_data.columns.get_loc("value")
    cols = list(edge_data.columns.values)
    swap = cols[val_loc]
    cols[val_loc] = cols[cur_loc]
    cols[cur_loc] = swap
    edge_data = edge_data.reindex(columns=cols)

    # Account for passed in node dataset
    if node_data is not None:
        node_dataset = hv.Dataset(node_data, 'index')
        chord = hv.Chord((edge_data, node_dataset), label=title).select(value=(5, None))
    else:
        chord = hv.Chord(edge_data, label=title)
        label_col = 'index'

    # Draw the desired graph
    if draw_labels is True:
        chord.opts(
            hv.opts.Chord(
                cmap=cmap, edge_cmap=cmap,
                edge_color=hv.dim(edge_color).str(),
                node_color=hv.dim(node_color).str(),
                labels=label_col,
            ))
    else:
        chord.opts(
            hv.opts.Chord(cmap=cmap, edge_cmap=cmap,
                          edge_color=hv.dim(edge_color).str(),
                          node_color=hv.dim(node_color).str()
                          ))
    c = hv.render(chord, backend='bokeh')
    return c


# Draw a simple circular node graph of g. Edges are straight lines with no weight information.
# Not as useful as graph g, but it is easier to make.
def plot_netgraph(g, ax=None, **kwargs):
    ax = get_axes(ax)
    node_layout = kwargs.setdefault('node_layout', nx.circular_layout(g))
    nx.draw(g, pos=node_layout,
            with_labels=True,
            arrows=True,
            **kwargs)
    return ax


# Draw arc diagram of graph g. Very handy for visualizing where motifs are in time.
# weight_attr is the edge attribute of graph g that has the edge weight information.
# node_labels should be made with the to_node_dict() function in graphutils.
def plot_arcgraph(g, ax=None,
                  weight_attr='value',
                  node_order=None,
                  node_labels=None,
                  node_color='w',
                  node_size=50.,
                  node_positions=None,
                  edge_width=5.,
                  edge_color=None,
                  **kwargs
                  ):
    adj_G = np.tril(nx.to_numpy_array(g, weight=weight_attr))
    vertical_shift = kwargs.setdefault('vertical_shift', 1 + (node_size / 100))
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


# Show a specific chord plot or all the matplotlib plots. Blocks other processes, so stick it at the end of a process.
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
    node_data = graph.to_node_dataframe(G)
    node_dict = graph.to_node_dict(G)
    plot_netgraph(G)
    plot_arcgraph(G, node_labels=node_dict, node_size=35.)
    c = plot_chordgraph(G, size=200, node_data=node_data, label_col='label')
    show(c)
    show()
    return


if __name__ == '__main__':
    main()
