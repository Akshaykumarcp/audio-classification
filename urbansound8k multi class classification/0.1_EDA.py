import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display

file_name = "dataset\\fold1\\7383-3-1-0.wav"

# plot wave form
plt.figure(figsize=(14,5))
data, sample_rate = librosa.load(file_name)
librosa.display.waveshow(data, sr=sample_rate)
ipd.Audio(file_name)
plt.show()

data # 1 dim data, mono audio type
"""
array([ 1.4551915e-10, -8.7311491e-11, -1.1641532e-10, ...,
        3.6435030e-04,  2.6052771e-04, -2.1291785e-04], dtype=float32) """

sample_rate # 22050
"""
Sample rate is the number of samples per second that are taken of a waveform to create a discete digital signal.
The higher the sample rate, the more snapshots you capture of the audio signal.
The audio sample rate is measured in kilohertz (kHz) and it determines the range of frequencies captured in digital audio. """

# same using scipy lib
from scipy.io import wavfile as wav

wave_sample_rate, wav_audio = wav.read(file_name)

wave_sample_rate # 44100

wav_audio # 2 dim data, sterio audio type
"""
array([[  0,   0],
       [  0,   0],
       [  0,   0],
       ...,
       [  6,  -5],
       [  7, -17],
       [ -5, -21]], dtype=int16) """

## plot 2 dim data, sterio audio type

plt.figure(figsize=(12,4))
plt.plot(wav_audio)
plt.show()

import pandas as pd

metadata = pd.read_csv("dataset/UrbanSound8K.csv")

metadata.head()
"""
      slice_file_name    fsID  start        end  salience  fold  classID             class
0    100032-3-0-0.wav  100032    0.0   0.317551         1     5        3          dog_bark
1  100263-2-0-117.wav  100263   58.5  62.500000         1     5        2  children_playing
2  100263-2-0-121.wav  100263   60.5  64.500000         1     5        2  children_playing
3  100263-2-0-126.wav  100263   63.0  67.000000         1     5        2  children_playing
4  100263-2-0-137.wav  100263   68.5  72.500000         1     5        2  children_playing """

## check data imbalance
metadata['class'].value_counts()
"""
dog_bark            1000
children_playing    1000
air_conditioner     1000
street_music        1000
engine_idling       1000
jackhammer          1000
drilling            1000
siren                929
car_horn             429
gun_shot             374
Name: class, dtype: int64 """

