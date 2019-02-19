from pathlib import Path
import librosa as lb
import networkx as nx
import numpy as np
import unittest
import fileutils


class BeatLengthTest(unittest.TestCase):
    def setUp(self):
        # Read in the audio
        name = 't1'
        directory = "../bin/"
        audio, fs = fileutils.load_audio(name, audio_dir=directory)

        # Identify beats in the audio
        tempo, beats = lb.beat.beat_track(y=audio, sr=fs, units='time')
        beats = np.concatenate([[0], beats, [audio.shape[0]]])

        # Construct graph from beats
        G = nx.DiGraph(tempo=tempo)
        for i in range(0, beats.shape[0] - 1):
            G.add_node(i, start=beats[i], end=beats[i + 1], length=(beats[i + 1] - beats[i]))

        # Create randomly-weighted edges for segments with exactly the same length
        for i in range(0, G.number_of_nodes()):
            for j in range(0, G.number_of_nodes()):
                if i <= j:  # No self loops
                    continue
                if G.node[i]['length'] == G.node[j]['length']:
                    e_weight = np.random.randn()
                    G.add_edge(i, j, value=e_weight)
        self.G = G

    def test_graph_has_strongly_connected_components(self):
        self.setUp()
        scc_count = nx.number_strongly_connected_components(self.G)
        self.assertGreater(scc_count, 0, 'Graph contains no strongly connected components')

    def test_graph_has_isolated_components(self):
        self.setUp()
        scc_G = nx.condensation(self.G)
        scc_G_nodes = nx.number_of_nodes(scc_G)
        scc_G_comp = nx.number_strongly_connected_components(scc_G)
        self.assertEqual(scc_G_comp, scc_G_nodes, 'Connected Components are not isolated')


if __name__ == '__main__':
    unittest.main()
