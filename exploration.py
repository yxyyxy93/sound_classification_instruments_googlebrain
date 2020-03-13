import numpy as np
import pandas as pd
import os
import glob
import librosa
import librosa.display
import re
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import time
import collections
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Store class names in array
class_names = ['bass', 'brass', 'flute', 'guitar',
               'keyboard', 'mallet', 'organ', 'reed',
               'string', 'synth_lead', 'vocal']
# Store source names in array
source_names = ['acoustic', 'electronic', 'synthetic']

# Pick a random  wave file from the dataset
bass_file = '/Users/nadimkawwa/Desktop/Udacity/MLEND/Capstone/nsynth-valid/audio/bass_electronic_018-047-075.wav'
brass_file = '/Users/nadimkawwa/Desktop/Udacity/MLEND/Capstone/nsynth-valid/audio/brass_acoustic_006-031-050.wav'
flute_file = '/Users/nadimkawwa/Desktop/Udacity/MLEND/Capstone/nsynth-valid/audio/flute_synthetic_000-035-127.wav'
guitar_file = '/Users/nadimkawwa/Desktop/Udacity/MLEND/Capstone/nsynth-valid/audio/guitar_acoustic_010-086-100.wav'
keyboard_file = '/Users/nadimkawwa/Desktop/Udacity/MLEND/Capstone/nsynth-valid/audio/keyboard_acoustic_004-041-100.wav'
mallet_file = '/Users/nadimkawwa/Desktop/Udacity/MLEND/Capstone/nsynth-valid/audio/mallet_acoustic_056-065-050.wav'

sample_files = [bass_file, brass_file, flute_file, guitar_file, keyboard_file]
# show 1 d signal

for i in sample_files:
    plt.figure(figsize=(10, 4))
    y, sr = librosa.load(i)
    plt.figure()
    plt.subplot(1, 1, 1)
    librosa.display.waveplot(y, sr=sr)
    plt.title('Monophonic')

plt.show()
