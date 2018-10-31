import numpy as np
from scipy import signal
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

r, sound = read('./main/bin/t1.wav', mmap=False)

f, t, Zxx = signal.stft(sound, r, nperseg=1000)
print(r)
print(sound.shape)
print(sound.shape[0])
print(Zxx.shape)
print(f.shape)
print(t.shape)

peaks = signal.find_peaks(np.abs(Zxx[0]))
print(peaks)

plt.figure()
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=1)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


