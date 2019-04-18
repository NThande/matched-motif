import time

import numpy as np

import analyzer
import config as cfg
import explots
import fileutils
import motifutils as motif
import visutils


def _analysis(analysis_name, audio, fs, length, methods, name='audio', show_plot=(),
              k=cfg.N_ClUSTERS, title_hook='{}', threshold=cfg.K_THRESH):
    G_dict = {}
    results_dict = {}

    if not isinstance(methods, tuple) and not isinstance(methods, list):
        methods = (methods,)

    for m in methods:
        if analysis_name == 'Segmentation':
            starts, ends, motif_labels, G = analyzer.analyze(audio, fs,
                                                             k_clusters=k,
                                                             seg_length=length,
                                                             seg_method=m,
                                                             threshold=threshold)

        elif analysis_name == 'Similarity':
            starts, ends, motif_labels, G = analyzer.analyze(audio, fs, k,
                                                             seg_length=length,
                                                             similarity_method=m,
                                                             threshold=threshold)

        elif analysis_name == 'K-Means':
            starts, ends, motif_labels, G = analyzer.analyze(audio, fs, m,
                                                             seg_length=length,
                                                             cluster_method=analysis_name,
                                                             threshold=threshold)

        elif analysis_name == 'Clustering':
            starts, ends, motif_labels, G = analyzer.analyze(audio, fs, k,
                                                             seg_length=length,
                                                             cluster_method=m,
                                                             threshold=threshold)
        elif analysis_name == 'Threshold':
            starts, ends, motif_labels, G = analyzer.analyze(audio, fs, threshold=m)
        else:
            print("Unrecognized analysis name: {exp_name}".format(exp_name=analysis_name))
            return None, None

        G_dict[m] = G
        results = motif.pack_motif(starts, ends, motif_labels)
        results_dict[m] = results

        title_suffix = title_hook.format(m)
        explots.draw_results(audio, fs, results, show_plot,
                             G=G,
                             name=name,
                             title_hook=title_suffix,
                             draw_ref=(m is methods[0]),
                             num_groups=k)
    return results_dict, G_dict


def segmentation_analysis(audio, fs, length, name='audio', show_plot=(),
                          methods=('regular', 'onset'), k=cfg.N_ClUSTERS):
    results_dict, G_dict = _analysis('Segmentation', audio, fs, length, methods,
                                     name=name, show_plot=show_plot, k=k,
                                     title_hook='with {} segmentation')
    return results_dict, G_dict


def similarity_analysis(audio, fs, length, name='audio', show_plot=(),
                        methods=('match', 'shazam'), k=cfg.N_ClUSTERS):
    results_dict, G_dict = _analysis('Similarity', audio, fs, length, methods,
                                     name=name, show_plot=show_plot, k=k,
                                     title_hook='with {} similarity')
    return results_dict, G_dict


def k_means_analysis(audio, fs, length, name='audio', show_plot=(),
                     k_clusters=(cfg.N_ClUSTERS, cfg.N_ClUSTERS + 1)):
    results_dict, G_dict = _analysis('K-Means', audio, fs, length, methods=k_clusters,
                                     name=name, show_plot=show_plot,
                                     title_hook='with {}-means clustering')
    return results_dict, G_dict


def clustering_analysis(audio, fs, length, name='audio', show_plot=(),
                        methods=('kmeans', 'agglom', 'spectral'), k=cfg.N_ClUSTERS):
    results_dict, G_dict = _analysis('Clustering', audio, fs, length, methods,
                                     name=name, show_plot=show_plot, k=k,
                                     title_hook='with {} clustering')
    return results_dict, G_dict


def threshold_analysis(audio, fs, length, name='audio', show_plot=(), threshold=(0, 0.5, 1)):
    exp_name = 'Threshold'
    results_dict, G_dict = _analysis(exp_name, audio, fs, length, methods=threshold,
                                     name=name, show_plot=show_plot,
                                     title_hook='')
    # title_hook='with {} Threshold')
    return results_dict, G_dict


# Write out an entire results set
def write_results(audio, fs, name, out_dir, methods, results):
    for m in methods:
        obs_motifs = results[m]
        write_name = name + ' ' + str(m)
        write_motifs(audio, fs, write_name, out_dir, obs_motifs)


# Write out all identified motifs in a single analysis
def write_motifs(audio, fs, name, audio_dir, motifs):
    time_id = int(round(time.time() * 1000))
    motif_labels = motifs[cfg.LABEL_IDX]
    num_motifs = motif_labels.shape[0]
    motif_dict = dict.fromkeys(np.unique(motif_labels), 0)
    for i in range(num_motifs):
        motif_start = int(motifs[cfg.START_IDX, i] * fs)
        motif_end = int(motifs[cfg.END_IDX, i] * fs)
        this_motif = audio[motif_start:motif_end]

        this_instance = motif_dict[motif_labels[i]]
        motif_dict[motif_labels[i]] = motif_dict[motif_labels[i]] + 1
        this_name = "{name}_m{motif}_i{instance}_{id}".format(name=name,
                                                              motif=int(motif_labels[i]),
                                                              instance=this_instance,
                                                              id=time_id)

        fileutils.write_audio(this_motif, fs, this_name, audio_dir)
    return


def tune_length_with_audio(audio, fs):
    return cfg.SEGMENT_LENGTH * np.ceil(((audio.shape[0] / fs) / 30)).astype(int)


def main():
    name = 'avril'
    in_dir = './bin/test'
    out_dir = './bin/results'
    audio, fs = fileutils.load_audio(name, audio_dir=in_dir)
    # audio_labels = fileutils.load_labels(name, label_dir=in_dir)
    # Should be sensitive to the length of the track, as well as k
    # Perhaps length should be extended as song goes longer than 30 seconds;
    # 3 second = 30 seconds, 18 seconds = 3 min
    # length = tune_length_with_audio(audio, fs)
    length = cfg.SEGMENT_LENGTH

    # explots.draw_reference(audio, fs, audio_labels, name=name,
    #                show_plot=('motif',))
    # segmentation_analysis(audio, fs, length, num_motifs=3, name=name,
    #                         show_plot=('arc',))
    # k_means_analysis(audio, fs, length, name=name, k_clusters=(5, 25, 50),
    #                  show_plot=('motif',))
    thresh = 0
    results, G_set = threshold_analysis(audio, fs, length, name=name,
                                        show_plot=('motif', 'matrix'), threshold=thresh)
    visutils.show()
    return


if __name__ == '__main__':
    main()
