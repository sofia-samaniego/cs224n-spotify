#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data.
"""
import os
import json
import nltk
#nltk.download('punkt')

import numpy as np
import re
import pickle
import random

#from __future__ import print_function

#P_CASE = "CASE:"
#CASES = ["aa", "AA", "Aa", "aA"]
START_TOKEN = "<s>"
END_TOKEN   = "</s>"
UNK_TOKEN   = "<unk>"
PAD_TOKEN   = "<mask>"

EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.05
MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 1500

MPD_PATH = '/Users/sofiasf/Data224n/SpotifyPlaylists/data/'
MPD_DEBUG_PATH = '/Users/sofiasf/Data224n/SpotifyPlaylists/data_debug/'
GLOVE_PATH = '/Users/sofiasf/Data224n/Glove/'
DATA_PATH = '/Users/sofiasf/Documents/Stanford/Win18/cs224n/cs224n-spotify/Data'
#GLOVE_PATH = '/home/sofiasf/cs224n-spotify/Data'
#DATA_PATH = '/home/sofiasf/cs224n-spotify/Data'

def process_mpd(path):
    data = []
    filenames = os.listdir(path)
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            max_len_y = 0
            max_len_x = 0
            for playlist in mpd_slice['playlists']:
                #y = normalize_name(playlist['name']).split()
                y = nltk.word_tokenize(normalize_name(playlist['name']))#.split()
                x = []
                for track in playlist['tracks']:
                    #x.extend(normalize_name(track['track_name']).split())
                    x.extend(nltk.word_tokenize(normalize_name(track['track_name'])))#.split())
                if len(y) > max_len_y:
                    max_len_y = len(y)
                if len(x) > max_len_x:
                    max_len_x = len(x)
                data.append((x,y))
    return (data, max_len_x, max_len_y)

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@-]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def create_GloVe_embedding(path):
    vocabulary = {}
    E = []
    with open(os.path.join(GLOVE_PATH, 'pruned/top_vectors.txt')) as f:
        i = 0
        for line in f:
            values = line.split()
            vocabulary[values[0]] = i
            E.append(values[1:])
            i += 1
    print('Found %s word vectors.'%len(vocabulary))

    # Add special tokens
    vocabulary[UNK_TOKEN] = len(vocabulary)
    E.append(np.zeros(len(values[1:])))
    vocabulary[START_TOKEN] = len(vocabulary)
    E.append(np.zeros(len(values[1:])))
    vocabulary[END_TOKEN] = len(vocabulary)
    E.append(np.zeros(len(values[1:])))
    vocabulary[PAD_TOKEN] = len(vocabulary)
    E.append(np.zeros(len(values[1:])))
    print('Added 4 special characters: <unk>, <mask>, <s>, </s>')

    return (vocabulary, np.asarray(E, dtype = 'float32'))

def token_to_idx(data, vocabulary):
    null_idx = vocabulary[UNK_TOKEN]
    data_idx = []
    for x, y in data:
        x_idx = []
        for token in x:
            if token in vocabulary:
                x_idx.append(vocabulary[token])
            else:
                x_idx.append(null_idx)
        y_idx = []
        for token in y:
            if token in vocabulary:
                y_idx.append(vocabulary[token])
            else:
                y_idx.append(null_idx)
        data_idx.append((x_idx, y_idx))
    return data_idx

def data_to_file(data, name):
    fname = os.path.join(DATA_PATH, name)
    with open(fname, 'w') as f:
        for tracks, title in data:
            for token in title:
                f.write(str(token))
                f.write(' ')
            f.write('\t')
            for token in tracks:
                f.write(str(token))
                f.write(' ')
            f.write('\n')
    return

def load_and_preprocess_data(output = 'tokens_all.txt', debug = False):
    fname = os.path.join(DATA_PATH, output)
    if not os.path.isfile(fname):
        dir_mpd = MPD_DEBUG_PATH if debug else MPD_PATH
        print('Reading from {} ...'.format(dir_mpd))
        data, max_x, max_y = process_mpd(dir_mpd)
        print('Writing into {} ...'.format(fname))
        with open(fname, 'w') as f:
            for tracks, title in data:
                for token in title:
                    f.write(str(token))
                    f.write(' ')
                f.write('\t')
                for token in tracks:
                    f.write(str(token))
                    f.write(' ')
                f.write('\n')
    else:
        data = []
        max_x = 0
        max_y = 0
        with open(fname, 'r') as f:
            print('Reading from {} ...'.format(fname))
            for line in f:
                title, tracks = line.split('\t')
                title = title.split()
                tracks = tracks.split()
                data.append((tracks, title))
                if len(title) > max_y:
                    max_y = len(title)
                if len(tracks) > max_x:
                    max_x = len(tracks)

    print('Found %s playlists.' % len(data))
    vocabulary, E = create_GloVe_embedding(GLOVE_PATH)
    data_idx = token_to_idx(data, vocabulary)

    # Filter out data with <unk> in playlist title
    filtered_data = []
    for x, y in data_idx:
        if vocabulary[UNK_TOKEN] not in y:
            filtered_data.append((x,y))

    # Split into train, dev, test
    N = len(data_idx)

    num_dev = int(N*VALIDATION_SPLIT)
    num_test = int(N*TEST_SPLIT)
    num_train = N - num_dev - num_test

    indices = np.arange(N)
    np.random.seed(23)
    np.random.shuffle(indices)

    data_idx = [data_idx[indices[i]] for i in range(N)]
    data = [data[indices[i]] for i in range(N)]

    train = data_idx[:num_train]
    dev   = data_idx[num_train:-num_test]
    test  = data_idx[-num_test:]

    train_raw = data[:num_train]
    dev_raw   = data[num_train:-num_test]
    test_raw  = data[-num_test:]

    data_to_file(train, 'train_eow.txt')
    data_to_file(dev, 'dev_eow.txt')
    data_to_file(test, 'test_eow.txt')

    data_to_file(train_raw, 'train_eow_raw.txt')
    data_to_file(dev_raw, 'dev_eow_raw.txt')
    data_to_file(test_raw, 'test_eow_raw.txt')

    data_to_file([([max_x], [max_y])], 'max_lengths_eow.txt')
    return train, dev, test, train_raw, dev_raw, test_raw, max_x, max_y, E, vocabulary

def load_data(train_name, dev_name, test_name, max_lengths = None):
    max_x = None
    max_y = None

    vocabulary, E = create_GloVe_embedding(GLOVE_PATH)

    # Load train
    train = []
    train_path = os.path.join(DATA_PATH, train_name)
    with open(train_path,'r') as f:
        for line in f:
            y,x=line.split('\t')
            y = y.split()
            x = x.split()
            train.append((x,y))

    # Load dev
    dev = []
    dev_path = os.path.join(DATA_PATH, dev_name)
    with open(dev_path,'r') as f:
        for line in f:
            y,x=line.split('\t')
            y = y.split()
            x = x.split()
            dev.append((x,y))

    # Load test
    test = []
    test_path = os.path.join(DATA_PATH, test_name)
    with open(test_path,'r') as f:
        for line in f:
            y,x=line.split('\t')
            y = y.split()
            x = x.split()
            test.append((x,y))

    N = len(train) + len(test) + len(dev)
    print("Found {} playlists".format(N))
    print("Train size: {}".format(len(train)))
    print("Dev size: {}".format(len(dev)))
    print("Test size: {}".format(len(test)))

    if max_lengths is not None:
        max_lengths_path = os.path.join(DATA_PATH, max_lengths)
        with open(max_lengths_path,'r') as f:
            for line in f:
                max_y, max_x = line.strip().split('\t')
        max_x = int(max_x)
        max_y = int(max_y)
        print("Max playlist name: {}".format(max_y))
        print("Max number of words in tracklist: {}".format(max_x))

    return train, dev, test, E, vocabulary, max_x, max_y

if __name__ == '__main__':
    pass

