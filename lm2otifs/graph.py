"""Corpus-level lexical co-occurrence graph (Section 4.1).

Node index space: documents first (train, then valid, then test), then one node per
vocabulary token. Edges (Eq. in Sec. 4.1):

* word-word : PMI(i, j) over sliding windows, kept only if PMI > 0,
              symmetrised with max{PMI(i, j), PMI(j, i)};
* doc-word  : TF-IDF of the token in the document.

Leakage-free protocol (default, used for every number in the paper):
the vocabulary, the window statistics behind PMI and the document frequencies
behind IDF are computed on the TRAINING documents only and then frozen.
Validation / test documents enter the graph only through their own doc-word
edges, restricted to in-vocabulary tokens (out-of-vocabulary tokens are dropped).

``transductive=True`` reproduces the original TextGCN-style construction in which
windows and IDF are computed over train + valid + test documents; it is kept only
for the transductive-vs-leakage-free comparison of Appendix E.1.
"""
from collections import defaultdict
from math import log

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from tqdm import tqdm

_TOKENIZERS = {}


def get_tokenizer(name='google-bert/bert-base-uncased'):
    """The paper uses the (slow) BERT WordPiece tokenizer; GPT-2 is the Table 17 variant."""
    if name not in _TOKENIZERS:
        from transformers import AutoTokenizer
        kwargs = {'do_lower_case': True} if 'uncased' in name else {}
        if 'bert' in name:
            kwargs['use_fast'] = False
        _TOKENIZERS[name] = AutoTokenizer.from_pretrained(name, **kwargs)
    return _TOKENIZERS[name]


def _windows(token_lists, window_size):
    windows = []
    for words in token_lists:
        if len(words) <= window_size:
            windows.append(words)
        else:
            for i in range(len(words) - window_size + 1):
                windows.append(words[i: i + window_size])
    return windows


def build_components(train_texts, train_labels, val_texts, val_labels,
                     test_texts, test_labels, tokenizer='google-bert/bert-base-uncased',
                     window_size=20, ordered=True, transductive=False):
    """Pre-compute every edge group of the corpus graph.

    ordered=True  : pairs inside a window are counted in precedence order
                    (preceding, following) before symmetrisation (paper default);
    ordered=False : both orders are counted (the "-U" ablation of Table 18).
    """
    tok = get_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
    doc_texts = [str(t).strip() for t in train_texts + val_texts + test_texts]
    labels = list(train_labels) + list(val_labels) + list(test_labels)
    n_train, n_val, n_test = len(train_texts), len(val_texts), len(test_texts)
    num_docs = len(doc_texts)
    split = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test

    # vocabulary: tokens of the training documents only
    doc_tokens, vocab_set, vocab = [], set(), []
    for i, text in enumerate(tqdm(doc_texts, desc='tokenize')):
        ids = tok(text)['input_ids']
        doc_tokens.append(ids)
        if i < n_train:
            for w in ids:
                if w not in vocab_set:
                    vocab_set.add(w)
                    vocab.append(w)
    word_id = {w: i for i, w in enumerate(vocab)}

    stat_docs = doc_tokens if transductive else doc_tokens[:n_train]
    n_stat = len(stat_docs)

    # ---- word-word PMI edges ----
    windows = _windows(stat_docs, window_size)
    window_freq = defaultdict(int)
    for window in windows:
        for w in set(window):
            window_freq[w] += 1
    pair_count = defaultdict(int)
    for window in tqdm(windows, desc='co-occurrence'):
        for i in range(1, len(window)):
            for j in range(i):
                wi, wj = word_id.get(window[i]), word_id.get(window[j])
                if wi is None or wj is None or wi == wj:
                    continue
                pair_count[(wj, wi)] += 1          # (preceding, following)
                if not ordered:
                    pair_count[(wi, wj)] += 1
    n_win = len(windows)
    ww_r, ww_c, ww_w = [], [], []
    for (i, j), count in pair_count.items():
        pmi = log((count / n_win) /
                  (window_freq[vocab[i]] * window_freq[vocab[j]] / (n_win * n_win)))
        if pmi <= 0:
            continue
        ww_r.append(num_docs + i)
        ww_c.append(num_docs + j)
        ww_w.append(pmi)

    # ---- document frequencies for IDF ----
    doc_freq = defaultdict(int)
    for ids in stat_docs:
        for w in set(ids):
            doc_freq[w] += 1

    def doc_word_edges(doc_ids):
        rows, cols, ws = [], [], []
        for i in doc_ids:
            counts = defaultdict(int)
            for w in doc_tokens[i]:
                if w in word_id:          # OOV tokens are dropped
                    counts[word_id[w]] += 1
            for wid, tf in counts.items():
                idf = log(n_stat / (doc_freq.get(vocab[wid], 0) + 1e-8))
                rows.append(i)
                cols.append(num_docs + wid)
                ws.append(tf * idf)
        return rows, cols, ws

    test_ids = list(range(n_train + n_val, num_docs))
    return {
        'ww': (ww_r, ww_c, ww_w),
        'train_dw': doc_word_edges(range(n_train)),
        'val_dw': doc_word_edges(range(n_train, n_train + n_val)),
        'test_dw_by_doc': {i: doc_word_edges([i]) for i in test_ids},
        'number_nodes': num_docs + len(vocab),
        'num_docs': num_docs,
        'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
        'labels': labels,
        'split': split,
        'vocab': vocab,                  # tokenizer ids, position = word node index - num_docs
        'test_ids': test_ids,
        'config': {'tokenizer': tokenizer if isinstance(tokenizer, str) else type(tokenizer).__name__,
                   'window_size': window_size, 'ordered': ordered,
                   'transductive': transductive},
    }


EMPTY = ([], [], [])


def assemble_adj(number_nodes, *edge_groups):
    """Sparse adjacency from (rows, cols, weights) groups, symmetrised with max(A, A^T)."""
    row, col, weight = [], [], []
    for r, c, w in edge_groups:
        row.extend(r)
        col.extend(c)
        weight.extend(w)
    a = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    return a + a.T.multiply(a.T > a) - a.multiply(a.T > a)


def to_pyg(adj, x, device, edge_norm='raw'):
    a = adj.tocoo()
    edge_index = torch.stack([torch.from_numpy(a.row).long(),
                              torch.from_numpy(a.col).long()]).to(device)
    w = torch.from_numpy(a.data).float().to(device)
    if edge_norm == 'log1p':
        w = torch.log1p(w)
    elif edge_norm != 'raw':
        raise ValueError(edge_norm)
    return Data(x=x, edge_index=edge_index, edge_attr=w)


def graph_views(comp, ww=None):
    """Adjacencies used for training, model selection and batched testing.

    Leakage-free: training sees only train doc-word edges (val/test doc nodes are
    isolated); model selection adds validation edges; batched testing adds every
    test document at once. Transductive: one graph with all documents for everything.
    """
    ww = comp['ww'] if ww is None else ww
    n = comp['number_nodes']
    all_test = [comp['test_dw_by_doc'][i] for i in comp['test_ids']]
    full = assemble_adj(n, ww, comp['train_dw'], comp['val_dw'], *all_test)
    if comp['config'].get('transductive'):
        return full, full, full
    return (assemble_adj(n, ww, comp['train_dw']),
            assemble_adj(n, ww, comp['train_dw'], comp['val_dw']),
            full)


def single_doc_adj(comp, doc_id, ww=None):
    """Strict per-document protocol: the frozen train+val graph plus ONE test document."""
    ww = comp['ww'] if ww is None else ww
    return assemble_adj(comp['number_nodes'], ww, comp['train_dw'], comp['val_dw'],
                        comp['test_dw_by_doc'][doc_id])


def subset_edges(edges, idx):
    r, c, w = (np.asarray(v) for v in edges)
    return r[idx].tolist(), c[idx].tolist(), w[idx].tolist()
