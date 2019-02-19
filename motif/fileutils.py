from pathlib import Path
import librosa as lb
import pandas as pd

DEFAULT_AUDIO_DIR = './bin/labelled'
DEFAULT_FS = 22050


def load_audio(name, audio_dir=DEFAULT_AUDIO_DIR, audio_type='wav', **kwargs):
    audio_suffix = '.' + audio_type
    audio_path = Path(audio_dir) / (name + audio_suffix)
    audio, fs = lb.load(audio_path, **kwargs)
    return audio, fs


def load_labels(audio_name, label_dir=DEFAULT_AUDIO_DIR, suffix='_labels'):
    label_path = Path(label_dir) / (audio_name + suffix + '.csv')
    labels = pd.read_csv(label_path)
    return labels


def write_audio(audio, fs, name, audio_dir=DEFAULT_AUDIO_DIR):
    audio_path = Path(audio_dir) / name
    lb.output.write_wav(str(audio_path), audio, fs, norm=True)
