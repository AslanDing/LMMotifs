import os
import random
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from math import log
import sys

datasets = ['../data/raid/subset/train_subset.csv',
            '../data/raid/subset/val_subset.csv',
            '../data/raid/subset/test_subset.csv']

word_embeddings_dim = 300
word_vector_map = {}

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

train_data = pd.read_csv(datasets[0])
train_lines = train_data['generation'].values
train_labels = train_data['model'].values
train_size = len(train_lines)

val_data = pd.read_csv(datasets[1])
val_lines = val_data['generation'].values
val_labels = val_data['model'].values
val_size = len(val_lines)

test_data = pd.read_csv(datasets[2])
test_lines = test_data['generation'].values
test_labels = test_data['model'].values
test_size = len(test_lines)

doc_content_list = []
for line in train_lines:
    doc_content_list.append(line.strip())

for line in val_lines:
    doc_content_list.append(line.strip())

for line in test_lines:
    doc_content_list.append(line.strip())


# build vocab
word_freq = {}
word_set = set()
for doc_words in doc_content_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)


vocab_str = '\n'.join(vocab)
f = open('../data/raid/corpus/vocab.txt', 'w')
f.write(vocab_str)
f.close()



word_doc_list = {}
for i in range(len(doc_content_list)):
    doc_words = doc_content_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i




### train
row_x = []
col_x = []
data_x = []
for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = word_doc_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    train_size, word_embeddings_dim))

y = []
for i in range(train_size):
    doc_meta = train_labels[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    if 'human' in label:
        y.append(0)
    else:
        y.append(1)
y = np.array(y)


### test
row_tx = []
col_tx = []
data_tx = []
for i in range(train_size+val_size,train_size+val_size+test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = word_doc_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(train_size+val_size,train_size+val_size+test_size):
    doc_meta = word_doc_list[i + train_size + val_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    if 'human' in label:
        ty.append(0)
    else:
        ty.append(1)
ty = np.array(ty)
print(ty)


word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))
for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []
for i in range(train_size,train_size+val_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = word_doc_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)
    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))
row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)
allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size,train_size+val_size):
    doc_meta = word_doc_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    if 'human' in label:
        ally.append(0)
    else:
        ally.append(1)
for i in range(vocab_size):
    # one_hot = [0 for l in range(len(label_list))]
    ally.append(0)

ally = np.array(ally)
print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)


# ========================================================================

# word co-occurence with context windows
window_size = 20
windows = []

for doc_words in word_doc_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)

word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

# pmi as weights

num_window = len(windows)
for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

doc_word_freq = {}
for doc_id in range(len(word_doc_list)):
    doc_words = word_doc_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1


for i in range(len(word_doc_list)):
    doc_words = word_doc_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(word_doc_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))


dataset = 'raid'
# dump objects
f = open("../data/raid/corpus/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("../data/raid/corpus/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("../data/raid/corpus/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("../data/raid/corpus/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("../data/raid/corpus/ind.{}.vx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("../data/raid/corpus/ind.{}.vy".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("../data/raid/corpus/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()

