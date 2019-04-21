import numpy as np

import analyzer
import config as cfg
import exputils
import fileutils
import graphutils as graph
import match_filter
import motifutils as motif
import shazam
import visutils as vis


# Some specific plots that I need
def draw_segmentation_evolution(audio, fs):
    fig = vis.get_fig()
    fig.suptitle('Comparison of Segmentation Windows')

    ax_over = draw_super_axis(fig)
    ax_over.set_xlabel('Time (Seconds)')
    ax_over.tick_params(labelcolor='w', top=False, bottom=True, left=False, right=False)

    ax = fig.add_subplot(2, 2, 1)
    ax.set_title('Audio Waveform', fontsize=14)
    vis.plot_audio_waveform(audio, fs, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    audio_len = int(audio.shape[0] / fs)
    ax.set_xticks(np.arange(audio_len + 1, step=np.ceil(audio_len / 5)))

    methods = ('Regular', 'Onset', 'Beat')
    step = (2, 5, 3)
    s_idx = 0
    f_idx = 2
    length = 3
    for m in methods:
        thumb, similarity, segments, sim_matrix = match_filter.thumbnail(audio, fs,
                                                                         seg_method=m,
                                                                         length=length)

        ax = fig.add_subplot(2, 2, f_idx)
        ax.set_ylabel('Window #', fontsize=14)
        ax = vis.plot_window_overlap(segments, np.ones(segments.shape) * length, audio_len,
                                     tick_step=step[s_idx], ax=ax)
        ax.set_title(m, fontsize=14)
        ax.grid()
        s_idx += 1
        f_idx += 1

    fig.subplots_adjust(top=0.85, wspace=0.4, hspace=0.4)
    return fig


def draw_matrix_evolution(audio, fs, length, name='audio'):
    adjacency_list = []
    title_list = []
    _, _, segments, adjacency = analyzer.self_similarity(audio, fs, length, method='Match', seg_method='Beat')
    adjacency_list.append(adjacency)
    title_list.append('Default')

    adjacency_no_overlap = analyzer.remove_overlap(segments, np.copy(adjacency), length)
    adjacency_list.append(adjacency_no_overlap)
    title_list.append('No Overlap')

    adjacency_reweight = analyzer.add_time_distance(segments, np.copy(adjacency_no_overlap))
    adjacency_list.append(adjacency_reweight)
    title_list.append('Time Weight Added')

    adjacency_thresh = analyzer.topk_threshold(np.copy(adjacency_no_overlap), cfg.K_THRESH)
    adjacency_list.append(adjacency_thresh)
    title_list.append('With Threshold')

    fig = vis.get_fig()
    fig.suptitle('Self-Similarity Matrix Evolution')
    num_plots = len(adjacency_list)
    plots_per_row = 2
    plots_per_col = np.ceil(num_plots / plots_per_row)

    ax_over = draw_super_axis(fig)
    ax_over.set_xlabel('Window #')
    ax_over.set_ylabel('Window #')
    ax_over.tick_params(labelcolor='w', top=False, bottom=True, left=False, right=False)

    for i in range(1, num_plots + 1):
        ax = fig.add_subplot(plots_per_row, plots_per_col, i)
        adj = adjacency_list[i - 1]

        ax, image = vis.plot_matrix(adj, ax=ax)
        ax.set_title(title_list[i - 1], fontsize=16)

        if i == num_plots:
            fig.subplots_adjust(right=0.8, top=0.85, wspace=0.4, hspace=0.4)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(image, cax=cbar_ax, format='%2.2f')
            cbar.set_label('Similarity')
    return fig


def draw_matrix_arc_chord_demo(G, name, with_chord=False):
    fig = vis.get_fig()
    fig.suptitle('Self-Similarity Visualizations')

    ax = draw_super_axis(fig)
    ax.tick_params(labelcolor='w', top=False, bottom=True, left=False, right=False)

    ax = fig.add_subplot(1, 2, 1)
    adjacency = graph.graph_to_adjacency_matrix(G)
    ax, _ = vis.plot_matrix(adjacency, ax=ax)
    ax.set_xlabel('Time Windows')
    ax.set_ylabel('Time Windows')
    ax.set_title('Self-Similarity Matrix')

    ax = fig.add_subplot(1, 2, 2)
    arc_labels = graph.to_node_dict(G, node_attr=cfg.NODE_LABEL)
    ax = vis.plot_arcgraph(G,
                           node_size=20.,
                           node_color='k',
                           node_labels=arc_labels,
                           font_size=20,
                           ax=ax,
                           vertical_shift=2)
    ax.set_xlabel("Time Segments")
    ax.set_title('Arc Diagram')

    if with_chord:
        chord_labels = graph.to_node_dataframe(G)
        c = vis.plot_chordgraph(G,
                                node_data=chord_labels,
                                label_col=cfg.NODE_LABEL,
                                title='Chord Graph Visualization')
        vis.show(c)
    return fig


def draw_motif_group(audio, fs, results, methods, title, subplots=(2, 2), label_prefix = ''):
    fig = vis.get_fig()
    fig.suptitle(title)
    plots_per_row = subplots[0]
    plots_per_col = subplots[1]
    f_idx = 1

    ax = draw_super_axis(fig)
    ax.tick_params(labelcolor='w', top=False, bottom=True, left=False, right=False)
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Amplitude')

    for m in methods:
        ax = fig.add_subplot(plots_per_row, plots_per_col, f_idx)
        starts, ends, labels = motif.unpack_motif(results[m])
        vis.plot_audio_waveform(audio, fs, ax=ax)
        labels = labels.astype(int)
        vis.add_motif_labels(ax, starts, ends, labels, alpha=0.5)
        ax.set_title(label_prefix + str(m), fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('')
        f_idx += 1

    fig.subplots_adjust(top=0.85, wspace=0.4, hspace=0.4)
    return fig


def draw_simple_arc(G, with_labels=True, with_color=True, ax=None, node_size=50.):
    if ax is None:
        fig = vis.get_fig()
        ax = draw_super_axis(fig)
    if with_labels:
        arc_labels = graph.to_node_dict(G, node_attr=cfg.NODE_LABEL)
    else:
        arc_labels = None
    num_nodes = len(G.nodes())
    num_groups = cfg.N_ClUSTERS

    if with_color:
        group_color = np.zeros(num_nodes)
        for i in G.nodes():
            group_color[i] = G.nodes()[i][cfg.CLUSTER_NAME] / num_groups
    else:
        group_color = 'w'

    ax = vis.plot_arcgraph(G,
                           node_size=node_size,
                           node_order=range(0, num_nodes),
                           node_labels=arc_labels,
                           node_color=group_color,
                           font_size=16,
                           ax=ax)
    return ax


def draw_super_axis(fig):
    ax = fig.add_subplot(1, 1, 1)  # The big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    return ax


def main():
    name = 'Repeat'
    in_dir = './bin/test'
    audio, fs = fileutils.load_audio(name, audio_dir=in_dir)
    length = cfg.SEGMENT_LENGTH
    #
    # fig = draw_segmentation_evolution(audio, fs)
    # vis.save_fig(fig, './bin/graphs/', 'SEG_{audio_name}'.format(audio_name=name))
    #
    # thresh = 0.95
    thresh = 0.985
    starts, ends, labels, G = analyzer.analyze(audio, fs, seg_length=length, threshold=thresh, seg_method='beat')
    #
    # fig = vis.get_fig()
    # fig.suptitle('Arc Graph Clustering')
    # ax = draw_super_axis(fig)
    # ax = draw_simple_arc(G, with_labels=True, with_color=True, ax=ax)
    # vis.save_fig(fig, './bin/graphs/', 'ARC_{audio_name}_clustered'.format(audio_name=name))

    fig = draw_matrix_arc_chord_demo(G, name, with_chord=False)
    vis.save_fig(fig, './bin/graphs/', 'SSM2ARC_{audio_name}'.format(audio_name=name))
    #
    # name = 't3'
    # in_dir = './bin/test'
    # audio, fs = fileutils.load_audio(name, audio_dir=in_dir)
    #
    # fig = draw_matrix_evolution(audio, fs, length, name)
    # vis.save_fig(fig, './bin/graphs/', 'SSM_{audio_name}'.format(audio_name=name))

    # results = {}
    # methods = ('With Clustering', 'With Join')
    #
    # name = 'Avril'
    # in_dir = "./bin/test"
    # audio, fs = fileutils.load_audio(name, audio_dir=in_dir)
    # length = cfg.SEGMENT_LENGTH
    #
    # audio_labels = fileutils.load_labels(name, label_dir=in_dir)
    # ref_starts, ref_ends, ref_labels = motif.df_to_motif(audio_labels)
    # fig = vis.get_fig()
    # ax = fig.add_subplot(1, 1, 1)
    # ax = vis.plot_motif_segmentation(audio, fs, ref_starts, ref_ends, ref_labels, ax=ax)
    # fig.suptitle('Hand-Labelled Description of {}'.format(name))
    # vis.save_fig(fig, './bin/graphs/', 'IDEAL_{audio_name}'.format(audio_name=name))


    # starts, ends, labels, G = analyzer.analyze(audio, fs, seg_length=length, with_join=False)
    # this_result = motif.pack_motif(starts, ends, labels)
    # results['With Merging'] = this_result
    #
    # starts, ends, labels, G = analyzer.analyze(audio, fs, seg_length=length, with_join=True)
    # this_result = motif.pack_motif(starts, ends, labels)
    # results['With Join'] = this_result
    #
    # fig = draw_motif_group(audio, fs, results, methods=methods, title='Joining Motif Segments', subplots=(2, 1))
    # vis.save_fig(fig, './bin/graphs/', 'MOTIF_{audio_name}'.format(audio_name=name))


    # name = 't1'
    # in_dir = "./bin/"
    # audio, fs = fileutils.load_audio(name, audio_dir=in_dir)
    # pairs_hash, pairs, peaks = shazam.fingerprint(audio)
    #
    # sxx = shazam.stft(audio,
    #                   n_fft=cfg.WINDOW_SIZE,
    #                   win_length=cfg.WINDOW_SIZE,
    #                   hop_length=int(cfg.WINDOW_SIZE * cfg.OVERLAP_RATIO),
    #                   window='hann')
    # sxx = np.abs(sxx)
    #
    # fig = vis.get_fig()
    # ax = fig.add_subplot(1, 1, 1)
    # ax = vis.plot_stft_with_pairs(sxx, peaks, pairs, ax=ax)
    # fig.suptitle('Spectrogram With Peaks and Pairs')
    #
    # vis.save_fig(fig, './bin/graphs/', 'SHAZAM_{audio_name}'.format(audio_name=name))

    vis.show()

    return


if __name__ == '__main__':
    main()
