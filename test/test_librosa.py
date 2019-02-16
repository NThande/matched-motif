from pathlib import Path
import librosa as lb
import numpy as np


def test_librosa():
    # 1. Get the file path to the audio
    audio_dir = Path("C:/Users/nthan/PycharmProjects/autosampler/bin/")
    file_name = audio_dir / "t1.wav"

    # 2. Load the audio as a waveform y with sampling rate sr
    y, sr = lb.load(file_name)

    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 512

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = lb.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = lb.beat.beat_track(y=y_percussive, sr=sr)

    # Compute MFCC features from the raw signal
    mfcc = lb.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

    # And the first-order differences (delta features)
    mfcc_delta = lb.feature.delta(mfcc)

    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = lb.util.sync(np.vstack([mfcc, mfcc_delta]),
                                   beat_frames)

    # Compute chroma features from the harmonic signal
    chromagram = lb.feature.chroma_cqt(y=y_harmonic,
                                       sr=sr)

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = lb.util.sync(chromagram, beat_frames, aggregate=np.median)

    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
