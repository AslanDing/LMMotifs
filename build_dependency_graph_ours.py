import os
import stanza
import pickle
import sys
import random
import numpy as np
from tqdm import tqdm
import json

# change your mkdir path of stanfordnlp model
nlp = stanza.Pipeline(lang='en', model_dir='./data/stanza_model',use_gpu=True,device='cuda:6')
if len(sys.argv) < 1:
    sys.exit("Use: python build_dependency_graph.py <dataset>")

# dataset = 'writing_claude-3-opus-20240229'#sys.argv[1]
# dataset = 'writing_gpt-3.5-turbo'#sys.argv[1]
# dataset = 'writing_gpt-4'#sys.argv[1]
# dataset = 'writing_gpt-4o'#sys.argv[1]
# dataset = 'writing_gpt-4o-mini'#sys.argv[1]
# dataset = 'writing_davinci'#sys.argv[1]
dataset = 'pubmed_gpt-4-1'#sys.argv[1]

# load pre-trained word embeddings
word_embeddings_dim = 300
word_embeddings = {}

with open('./data/glove.6B.' + str(word_embeddings_dim) + 'd.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        data = line.split()
        word_embeddings[str(data[0])] = list(map(float, data[1:]))

# load json
path = f'../data/{dataset}.raw_data.json'
with open(path) as fp:
    data = json.load(fp)

original = data['original']
sampled = data['sampled']

# load label list
# doc_name_list = []
# doc_train_list = []
# doc_test_list = []
# with open('../data/' + dataset + '.txt', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         doc_name_list.append(line.strip())
#         temp = line.split("\t")
#
#         if temp[1].find('test') != -1:
#             doc_test_list.append(line.strip())
#         elif temp[1].find('train') != -1:
#             doc_train_list.append(line.strip())

# load content
# doc_content_list = []
# with open('data/corpus/' + dataset + '.clean.txt', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         doc_content_list.append(line.strip())

# map and shuffle
# train_ids = []
# for train_name in doc_train_list:
#     train_id = doc_name_list.index(train_name)
#     train_ids.append(train_id)
# random.shuffle(train_ids)
#
# test_ids = []
# for test_name in doc_test_list:
#     test_id = doc_name_list.index(test_name)
#     test_ids.append(test_id)
# random.shuffle(test_ids)
#
# ids = train_ids + test_ids
#
# # train + test
# shuffle_doc_name_list = []
# shuffle_doc_words_list = []
# for i in ids:
#     shuffle_doc_name_list.append(doc_name_list[int(i)])
#     shuffle_doc_words_list.append(doc_content_list[int(i)])
#
# # build label list
# label_set = set()
# for doc_meta in shuffle_doc_name_list:
#     temp = doc_meta.split('\t')
#     label_set.add(temp[2])
# label_list = list(label_set)

# train_size = len(train_ids)
# test_size = len(test_ids)


def build_dptree(txt_lists):
    # build dptree data
    trees = []
    labels = []
    features = []

    for i,content in tqdm(enumerate(txt_lists)):
        # content = shuffle_doc_words_list[i]
        doc = nlp(content)
        doc_dt = []
        doc_feature = []
        # trees features
        bias = 0

        for sentence in doc.sentences:
            word_num = 0
            for word in sentence.words:
                word_num += 1
                doc_feature.append(word_embeddings[word.text] if word.text in word_embeddings else np.random.uniform(-0.01, 0.01, word_embeddings_dim))
                if word.head == 0:
                    deptree = {'id': int(word.id) + bias,
                               'text': word.text,
                               'parent': word.head}
                else:
                    deptree = {'id': int(word.id) + bias,
                               'text': word.text,
                               'parent': word.head + bias}
                doc_dt.append(deptree)
            bias += word_num
        trees.append(doc_dt)
        features.append(doc_feature)

    return trees, features

print('building graphs for training')
train_tree, train_feature = build_dptree(original)
print('building graphs for test')
test_tree, test_feature = build_dptree(sampled)
data = {'original_tree': train_tree,
        'original_feature': train_feature,
        'sampled_tree': test_tree,
        'sampled_feature': test_feature,
        }
root_path = './data/ours/trees/'
if not os.path.exists(root_path):
    os.makedirs(root_path)
fout = open(root_path + dataset + '.pickle', 'wb')
pickle.dump(data, fout)
