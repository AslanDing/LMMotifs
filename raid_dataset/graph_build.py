import torch
import os.path

import networkx as nx
import scipy.sparse as sp
from math import log
from dataset import TextDataset
from collections import defaultdict
import pandas as pd
# from utils import get_corpus_path
from os.path import join, exists
from tqdm import tqdm
from transformers import BertTokenizer

model_name = 'gchhablani/bert-base-cased-finetuned-sst2'   # Choose your preferred model name
# model_name = 'gchhablani/bert-base-cased-finetuned-sst2' # 0.92
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)


def build_text_graph_dataset1(dir, window_size=20):
    # if "small" in dataset or "presplit" in dataset or 'sentiment' in dataset:
    #     dataset_name = "_".join(dataset.split("_")[:-1])
    # else:
    #     dataset_name = dataset
    if os.path.exists(f'{dir}/raid_subset.pth'):
        textdataset = torch.load(f'{dir}/raid_subset.pth')
        return textdataset
    else:
        pass

    datasets = [f'{dir}/train_subset.csv',
                f'{dir}/val_subset.csv',
                f'{dir}/test_subset.csv']
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

    train_split = {}
    doc_list = []
    count = 0
    for line in train_lines:
        doc_list.append(line.strip())
        train_split[count]='train'
        count+=1
    for line in val_lines:
        doc_list.append(line.strip())
        train_split[count]='val'
        count += 1
    for line in test_lines:
        doc_list.append(line.strip())
        train_split[count]='test'
        count += 1

    labels = []
    for line in train_labels:
        if 'human' in line:
            labels.append(0)
        else:
            labels.append(1)
    for line in val_labels:
        if 'human' in line:
            labels.append(0)
        else:
            labels.append(1)
    for line in test_labels:
        if 'human' in line:
            labels.append(0)
        else:
            labels.append(1)

    assert len(labels) == len(doc_list)

    word_freq,doc_lists_ = get_vocab1(doc_list)
    vocab = list(word_freq.keys())
    if not exists(dir + '/vocab.txt'):
        vocab_txt = [ str(c) for c in vocab]
        vocab_str = '\n'.join(vocab_txt)
        f = open(dir + '/vocab.txt', 'w')
        f.write(vocab_str)
        f.close()
    words_in_docs, word_doc_freq = build_word_doc_edges1(doc_lists_)
    word_id_map = {word: i for i, word in enumerate(vocab)}

    sparse_graph = build_edges1(doc_lists_, word_id_map, vocab, word_doc_freq, window_size)
    docs_dict = {i: doc for i, doc in enumerate(doc_lists_)}
    textdataset = TextDataset('raid', sparse_graph, labels, vocab, word_id_map, docs_dict, None,
                       train_test_split=train_split)
    torch.save(textdataset,f'{dir}/raid_subset.pth')
    return textdataset

def build_edges1(doc_list, word_id_map, vocab, word_doc_freq, window_size=20):
    # constructing all windows
    windows = []
    for doc_words in tqdm(doc_list):
        words = doc_words #.split()
        doc_length = len(words)
        if doc_length <= window_size:
            windows.append(words)
        else:
            for i in range(doc_length - window_size + 1):
                window = words[i: i + window_size]
                windows.append(window)
    # constructing all single word frequency
    word_window_freq = defaultdict(int)
    for window in tqdm(windows):
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
    # constructing word pair count frequency
    word_pair_count = defaultdict(int)
    for window in tqdm(windows):
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_j = window[j]
                word_i_id = word_id_map[word_i]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_count[(word_i_id, word_j_id)] += 1
                word_pair_count[(word_j_id, word_i_id)] += 1
    row = []
    col = []
    weight = []

    # pmi as weights
    num_docs = len(doc_list)
    num_window = len(windows)
    for word_id_pair, count in tqdm(word_pair_count.items()):
        i, j = word_id_pair[0], word_id_pair[1]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(num_docs + i)
        col.append(num_docs + j)
        weight.append(pmi)

    # frequency of document word pair
    doc_word_freq = defaultdict(int)
    for i, doc_words in enumerate(doc_list):
        words = doc_words#.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = (i, word_id)
            doc_word_freq[doc_word_str] += 1

    for i, doc_words in enumerate(doc_list):
        words = doc_words #.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            word_id = word_id_map[word]
            freq = doc_word_freq[(i, word_id)]
            row.append(i)
            col.append(num_docs + word_id)
            idf = log(1.0 * num_docs /
                      (word_doc_freq[vocab[word_id]]+1E-8))
            weight.append(freq * idf)
            doc_word_set.add(word)

    number_nodes = num_docs + len(vocab)
    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    return adj

def build_edges(doc_list, word_id_map, vocab, word_doc_freq, window_size=20):
    # constructing all windows
    windows = []
    for doc_words in tqdm(doc_list):
        words = doc_words.split()
        doc_length = len(words)
        if doc_length <= window_size:
            windows.append(words)
        else:
            for i in range(doc_length - window_size + 1):
                window = words[i: i + window_size]
                windows.append(window)
    # constructing all single word frequency
    word_window_freq = defaultdict(int)
    for window in tqdm(windows):
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
    # constructing word pair count frequency
    word_pair_count = defaultdict(int)
    for window in tqdm(windows):
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_j = window[j]
                word_i_id = word_id_map[word_i]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_count[(word_i_id, word_j_id)] += 1
                word_pair_count[(word_j_id, word_i_id)] += 1
    row = []
    col = []
    weight = []

    # pmi as weights
    num_docs = len(doc_list)
    num_window = len(windows)
    for word_id_pair, count in tqdm(word_pair_count.items()):
        i, j = word_id_pair[0], word_id_pair[1]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(num_docs + i)
        col.append(num_docs + j)
        weight.append(pmi)

    # frequency of document word pair
    doc_word_freq = defaultdict(int)
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = (i, word_id)
            doc_word_freq[doc_word_str] += 1

    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            word_id = word_id_map[word]
            freq = doc_word_freq[(i, word_id)]
            row.append(i)
            col.append(num_docs + word_id)
            idf = log(1.0 * num_docs /
                      word_doc_freq[vocab[word_id]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    number_nodes = num_docs + len(vocab)
    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    return adj

def get_vocab1(text_list):
    doc_lists = []
    word_freq = defaultdict(int)
    for doc_words in tqdm(text_list):

        encoded_data = tokenizer(doc_words)
        input_ids = encoded_data["input_ids"]
        doc_lists.append(input_ids)
        words = input_ids
        for word in words:
            word_freq[word] += 1
    return word_freq,doc_lists

def get_vocab(text_list):
    word_freq = defaultdict(int)
    for doc_words in text_list:
        words = doc_words.split()
        for word in words:
            word_freq[word] += 1
    return word_freq

def build_word_doc_edges1(doc_list):
    # build all docs that a word is contained in
    words_in_docs = defaultdict(set)
    for i, doc_words in enumerate(doc_list):
        words = doc_words#.split()
        for word in words:
            words_in_docs[word].add(i)

    word_doc_freq = {}
    for word, doc_list in words_in_docs.items():
        word_doc_freq[word] = len(doc_list)

    return words_in_docs, word_doc_freq

def build_word_doc_edges(doc_list):
    # build all docs that a word is contained in
    words_in_docs = defaultdict(set)
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:
            words_in_docs[word].add(i)

    word_doc_freq = {}
    for word, doc_list in words_in_docs.items():
        word_doc_freq[word] = len(doc_list)

    return words_in_docs, word_doc_freq


if __name__ == "__main__":
    dir ='../data/raid/subset'
    textdataset = build_text_graph_dataset1(dir, 20)
    # print(textdataset[0])
    print("xx")



