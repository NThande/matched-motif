import holoviews as hv
import pandas as pd
import bokeh as bk
from bokeh.plotting import show
from bokeh.sampledata.les_mis import data


def test_chordgraph():
    hv.extension('bokeh')
    hv.output(size=200)

    links = pd.DataFrame(data['links'])
    print(links.head(3))

    c = hv.render(hv.Chord(links), backend='bokeh')
    print(type(c))
    bk.plotting.show(c)

    nodes = hv.Dataset(pd.DataFrame(data['nodes']), 'index')
    nodes.data.head()

    chord = hv.Chord((links, nodes)).select(value=(5, None))
    chord.opts(hv.opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=hv.dim('source').str(),
                          labels='name', node_color=hv.dim('index').str()))
    bk.plotting.show(hv.render(chord, backend='bokeh'))
