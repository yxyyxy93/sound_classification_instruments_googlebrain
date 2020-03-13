#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import scipy
import time
import collections
import itertools
import librosa
import pickle
from pathlib import Path


def feature_extract(file):
    """
    Define function that takes in a file an returns features in an array
    """
    # get wave representation
    y, sr = librosa.load(file, duration=2.97)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    return ps


def instrument_code(filename):
    """
    Function that takes in a filename and returns instrument based on naming convention
    """
    class_names = ['bass', 'brass', 'flute', 'guitar',
                   'keyboard', 'mallet', 'organ', 'reed',
                   'string', 'synth_lead', 'vocal']
    for name in class_names:
        if name in filename:
            return class_names.index(name)
    else:
        return None


if __name__ == "__main__":
    data_folder = Path("E:/other_projects/sound_classification_instrument/")
    # directory to training data and json file
    train_dir = data_folder / 'nsynth-train' / 'audio'
    # directory to training data and json file
    valid_dir = data_folder / 'nsynth-valid' / 'audio'
    # directory to training data and json file
    test_dir = data_folder / 'nsynth-test' / 'audio'
    '''
    read the raw json files as given in the training set
    '''
    df_train_raw = pd.read_json(path_or_buf='nsynth-train/examples.json', orient='index')
    # Get a count of instruments in ascending order
    n_class_train = df_train_raw['instrument_family'].value_counts(ascending=True)
    print('n_class of train:', n_class_train)
    # Sample n files
    df_train_sample = df_train_raw.groupby('instrument_family',
                                           as_index=False,  # group by instrument family
                                           group_keys=False).apply(lambda df: df.sample(2000))  # number of samples
    # drop the synth_lead from the training dataset
    df_train_sample = df_train_sample[df_train_sample['instrument_family'] != 9]
    # save the train file index as list
    filenames_train = df_train_sample.index.tolist()
    # save the list to a pickle file
    with open(data_folder / 'dataset/filenames_train.pickle', 'wb') as f:
        pickle.dump(filenames_train, f)
    start_train = time.time()
    # create dictionary to store all test features
    dict_train = {}
    print('samples number of training', len(filenames_train))
    # loop over every file in the list
    i = 0
    D = []
    for file in filenames_train:
        # extract the features
        ps = feature_extract(train_dir / (file + '.wav'))  # specify directory and .wav
        code = instrument_code(file)
        D.append((ps, code))
        i += 1
        if i % 100 == 0:
            print(i, '/', len(filenames_train))
    end_train = time.time()
    print(end_train - start_train)
    with open(data_folder / 'dataset/train.pickle', 'wb') as f:
        pickle.dump(D, f)
    '''
    extract the filenames from the validation dataset
    '''
    df_valid = pd.read_json(path_or_buf=data_folder / 'nsynth-valid/examples.json', orient='index')
    # save the train file index as list
    filenames_valid = df_valid.index.tolist()
    # save the list to a pickle file
    D = []
    for file in filenames_valid:
        # extract the features
        ps = feature_extract(valid_dir / (file + '.wav'))  # specify directory and .wav
        code = instrument_code(file)
        D.append((ps, code))
        i += 1
        if i % 100 == 0:
            print(i, '/', len(filenames_valid))
    end_train = time.time()
    print(end_train - start_train)
    with open(data_folder / 'dataset/valid.pickle', 'wb') as f:
        pickle.dump(D, f)
    '''
    extract the filenames from the testing dataset
    '''
    df_test = pd.read_json(path_or_buf=data_folder / 'nsynth-test/examples.json', orient='index')
    # save the train file index as list
    filenames_test = df_test.index.tolist()
    for file in filenames_test:
        # extract the features
        ps = feature_extract(test_dir / (file + '.wav'))  # specify directory and .wav
        code = instrument_code(file)
        D.append((ps, code))
        i += 1
        if i % 100 == 0:
            print(i, '/', len(filenames_test))
    end_train = time.time()
    print(end_train - start_train)
    with open(data_folder / 'dataset/test.pickle', 'wb') as f:
        pickle.dump(D, f)