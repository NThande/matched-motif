# motif-discovery
Matched-Filter Musical Motif Discovery

This is a system that uses the Matched Filter to find all repeating parts in a song.

This was used to create the Princeton University senior thesis paper, "Matched-Filter Musical Motif Discovery".

This repository is for documentation purposes; future work is not planned. 

# Usage

To find motifs, simply load in audio and call:

```
starts, ends, labels, motif_graph = analyzer.analyze(audio, fs)
```

There are many options for changing how the input is calculated. 

More info to come. 
