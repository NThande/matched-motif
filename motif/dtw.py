import librosa as lb
import fileutils
import fastdtw
from scipy.spatial.distance import euclidean
import time


def main():
    name = 't3_train'
    directory = "./bin/labelled"
    fs = 4000
    audio, fs = fileutils.load_audio(name, audio_dir=directory, sr=fs)

    x = audio[0:3 * fs]
    y = audio[1 * fs:4 * fs]

    # start = time.time()
    # D, wp = lb.sequence.dtw(x, y, metric='euclidean', subseq=True, backtrack=True)
    # end = time.time()
    # print("Standard Time: {}".format(end - start))

    start = time.time()
    D, wp = fastdtw.fastdtw(x, y, dist=euclidean)
    end = time.time()

    print("Fast Time: {}".format(end - start))
    # print(wp)


if __name__ == '__main__':
    main()
