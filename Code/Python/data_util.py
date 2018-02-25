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
#START_TOKEN = "<s>"
#END_TOKEN = "</s>"
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.05
MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 1500

MPD_PATH = '/Users/sofiasf/Data224n/SpotifyPlaylists/data/'
GLOVE_PATH = '/Users/sofiasf/Data224n/Glove/'
DATA_PATH = '/Users/sofiasf/Documents/Stanford/Win18/cs224n/cs224n-spotify/Data'

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
    print('Found %s playlists.' % len(data))
    return (data, max_len_x, max_len_y)

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@-]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def load_GloVe(path):
    vocabulary = {}
    E = []
    with open(os.path.join(GLOVE_PATH, 'glove.6B.100d.txt')) as f:
        i = 0
        for line in f:
            values = line.split()
            vocabulary[values[0]] = i
            E.append(values[1:])
            i += 1
    print('Found %s word vectors.'%len(vocabulary))
    E.append(np.zeros(len(values[1:])))
    return (vocabulary, np.asarray(E, dtype = 'float32'))

def token_to_idx(data, vocabulary):
    null_idx = len(vocabulary)
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

def load_and_preprocess_data(output = 'tokens_tracks.txt', process_mpd = False):
    fname = os.path.join(DATA_PATH, output)
    if not os.path.isfile(fname):
        data, max_x, max_y = process_mpd(MPD_PATH)
        with open(fname, 'w') as f:
            for tracks, title in data:
                for token in title:
                    f.write(token)
                    f.write(' ')
                f.write('\t')
                for token in tracks:
                    f.write(token)
                    f.write(' ')
                f.write('\n')
    else:
        data = []
        max_x = 0
        max_y = 0
        with open(fname, 'r') as f:
            for line in f:
                title, tracks = line.split('\t')
                title = title.split()
                tracks = tracks.split()
                data.append((tracks, title))
                if len(title) > max_y:
                    max_y = len(y)
                if len(tracks) > max_x:
                    max_x = len(x)

    vocabulary, E = loadGloVe(GLOVE_PATH)
    data_idx = token_to_idx(data, vocabulary)

    # Split into train, dev, test
    N = len(data_idx)

    num_dev = int(N*VALIDATION_SPLIT)
    num_test = int(N*TEST_SPLIT)
    num_train = N - num_dev - num_test

    indices = np.arange(N)
    np.random.seed(23)
    np.random.shuffle(indices)
    data_idx = data_idx[indices]
    data = data[indices]

    train = data_idx[:num_train]
    dev   = data_idx[(num_train+1):-num_test]
    test  = data_idx[-num_test:]


    train_raw = data[:num_train]
    dev_raw   = data[num_train:-num_test]
    test_raw  = data[-num_test:]

    return train, dev, test, train_raw, dev_raw, test_raw, max_x, max_y, E


if __name__ == '__main__':
    pass

# def load_embeddings(path):
    # embeddings = {}
    # with open(os.path.join(GLOVE_PATH, 'glove.6B.100d.txt')) as f:
        # for line in f:
            # values = line.split()
            # word = values[0]
            # coefs = np.asarray(values[1:], dtype = 'float32')
            # embeddings[word] = coefs
    # print('Found %s word vectors.'%len(embeddings))
    # return embeddings

# def create_embedding_matrix(embeddings):
    # word_to_idx = {token: idx for idx, token in enumerate(set(embeddings.keys()))}
    # E = np.zeros((len(word_to_idx) + 1, EMBEDDING_DIM))
    # for word, i in word_to_idx.items():
        # embedding_vector = embeddings.get(word)
        # if embedding_vector is not None:
            # E[i] = embedding_vector
    # return E, word_to_idx

# def to_date(epoch):
    # return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d")


# def casing(word):
    # if len(word) == 0: return word

    # # all lowercase
    # if word.islower(): return "aa"
    # # all uppercase
    # elif word.isupper(): return "AA"
    # # starts with capital
    # elif word[0].isupper(): return "Aa"
    # # has non-initial capital
    # else: return "aA"

# def normalize(word):
    # """
    # Normalize words that are numbers or have casing.
    # """
    # if word.isdigit(): return NUM
    # else: return word.lower()

# def featurize(embeddings, word):
    # """
    # Featurize a word given embeddings.
    # """
    # case = casing(word)
    # word = normalize(word)
    # case_mapping = {c: one_hot(FDIM, i) for i, c in enumerate(CASES)}
    # wv = embeddings.get(word, embeddings[UNK])
    # fv = case_mapping[case]
    # return np.hstack((wv, fv))

# class ModelHelper(object):
    # """
    # This helper takes care of preprocessing data, constructing embeddings, etc.
    # """
    # def __init__(self, tok2id, max_length):
        # self.tok2id = tok2id
        # self.START = [tok2id[START_TOKEN], tok2id[P_CASE + "aa"]]
        # self.END = [tok2id[END_TOKEN], tok2id[P_CASE + "aa"]]
        # self.max_length = max_length

    # def vectorize_example(self, sentence, labels=None):
        # sentence_ = [[self.tok2id.get(normalize(word), self.tok2id[UNK]), self.tok2id[P_CASE + casing(word)]] for word in sentence]
        # if labels:
            # labels_ = [LBLS.index(l) for l in labels]
            # return sentence_, labels_
        # else:
            # return sentence_, [LBLS[-1] for _ in sentence]

    # def vectorize(self, data):
        # return [self.vectorize_example(sentence, labels) for sentence, labels in data]

    # @classmethod
    # def build(cls, data):
        # # Preprocess data to construct an embedding
        # # Reserve 0 for the special NIL token.
        # tok2id = build_dict((normalize(word) for sentence, _ in data for word in sentence), offset=1, max_words=10000)
        # tok2id.update(build_dict([P_CASE + c for c in CASES], offset=len(tok2id)))
        # tok2id.update(build_dict([START_TOKEN, END_TOKEN, UNK], offset=len(tok2id)))
        # assert sorted(tok2id.items(), key=lambda t: t[1])[0][1] == 1
        # logger.info("Built dictionary for %d features.", len(tok2id))

        # max_length = max(len(sentence) for sentence, _ in data)

        # return cls(tok2id, max_length)

    # def save(self, path):
        # # Make sure the directory exists.
        # if not os.path.exists(path):
            # os.makedirs(path)
        # # Save the tok2id map.
        # with open(os.path.join(path, "features.pkl"), "w") as f:
            # pickle.dump([self.tok2id, self.max_length], f)

    # @classmethod
    # def load(cls, path):
        # # Make sure the directory exists.
        # assert os.path.exists(path) and os.path.exists(os.path.join(path, "features.pkl"))
        # # Save the tok2id map.
        # with open(os.path.join(path, "features.pkl")) as f:
            # tok2id, max_length = pickle.load(f)
        # return cls(tok2id, max_length)

# def load_and_preprocess_data(args):
    # logger.info("Loading training data...")
    # train = read_conll(args.data_train)
    # logger.info("Done. Read %d sentences", len(train))
    # logger.info("Loading dev data...")
    # dev = read_conll(args.data_dev)
    # logger.info("Done. Read %d sentences", len(dev))

    # helper = ModelHelper.build(train)

    # # now process all the input data.
    # train_data = helper.vectorize(train)
    # dev_data = helper.vectorize(dev)

    # return helper, train_data, dev_data, train, dev

# def load_embeddings(args, helper):
    # embeddings = np.array(np.random.randn(len(helper.tok2id) + 1, EMBED_SIZE), dtype=np.float32)
    # embeddings[0] = 0.
    # for word, vec in load_word_vector_mapping(args.vocab, args.vectors).items():
        # word = normalize(word)
        # if word in helper.tok2id:
            # embeddings[helper.tok2id[word]] = vec
    # logger.info("Initialized embeddings.")

    # return embeddings

# def build_dict(words, max_words=None, offset=0):
    # cnt = Counter(words)
    # if max_words:
        # words = cnt.most_common(max_words)
    # else:
        # words = cnt.most_common()
    # return {word: offset+i for i, (word, _) in enumerate(words)}


# def get_chunks(seq, default=LBLS.index(NONE)):
    # """Breaks input of 4 4 4 0 0 4 0 ->   (0, 4, 5), (0, 6, 7)"""
    # chunks = []
    # chunk_type, chunk_start = None, None
    # for i, tok in enumerate(seq):
        # # End of a chunk 1
        # if tok == default and chunk_type is not None:
            # # Add a chunk.
            # chunk = (chunk_type, chunk_start, i)
            # chunks.append(chunk)
            # chunk_type, chunk_start = None, None
        # # End of a chunk + start of a chunk!
        # elif tok != default:
            # if chunk_type is None:
                # chunk_type, chunk_start = tok, i
            # elif tok != chunk_type:
                # chunk = (chunk_type, chunk_start, i)
                # chunks.append(chunk)
                # chunk_type, chunk_start = tok, i
        # else:
            # pass
    # # end condition
    # if chunk_type is not None:
        # chunk = (chunk_type, chunk_start, len(seq))
        # chunks.append(chunk)
    # return chunks

# def test_get_chunks():
    # assert get_chunks([4, 4, 4, 0, 0, 4, 1, 2, 4, 3], 4) == [(0,3,5), (1, 6, 7), (2, 7, 8), (3,9,10)]
