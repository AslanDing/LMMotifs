"""Dataset loaders.

Every loader returns six lists

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

with label 0 = human-written (HGT) and 1 = machine-generated (MGT).
Training sets are capped at ``max_per_class`` documents per class (1000 in the paper).

Expected files under ``--data_root`` (see README for how to obtain them):

    hc3/{domain}_{train,valid,test}.csv          columns human_answers, chatgpt_answers
                                                 (built by scripts/preprocess_hc3.py)
    m4/{domain}_{model}_{train,valid,test}.csv   columns human_text, machine_text
    raid/{domain}_human_{split}.csv              column generation
    raid/{domain}_{model}_{split}.csv            column generation
    {LLMDIR}/{Yelp,Essay,Creative}_human.json    list of str, or list of {'prompt', 'essay'}
    {LLMDIR}/{Dataset}_{model-tag}.json          list of str
    pubmed/pubmed_{model}.json                   {"original": [...], "sampled": [...]}
"""
import json
import os

import numpy as np
import pandas as pd

SPLIT_SEED = 42
MAX_PER_CLASS = 1000

HC3_DOMAINS = ['open_qa', 'wiki_csai', 'medicine', 'finance']
M4_DOMAINS = ['wikihow', 'reddit', 'peerread', 'arxiv']
RAID_DOMAINS = ['recipes', 'books', 'poetry', 'reviews']
YEC_DATASETS = ['Yelp', 'Essay', 'Creative']

# short model name -> (directory, file tag) for Yelp / Essay / Creative
YEC_MODELS = {
    'sonnet': ('claude3-sonnet', 'claude-3-sonnet'),
    'opus': ('claude3-opus', 'claude-3-opus'),
    'gemini': ('Gemini', 'gemini-1.0-pro'),
    'gpt35': ('GPT-3.5', 'gpt-3.5-turbo'),
    'gpt4': ('GPT-4', 'gpt-4-turbo-preview'),
}


def _cap(lines, n):
    return lines[:n] if n else lines


def _pack(human, machine):
    return human + machine, [0] * len(human) + [1] * len(machine)


def _clean(seq):
    return [str(x).strip() for x in seq if isinstance(x, str) and str(x).strip()]


def load_hc3(root, domain, model=None, max_per_class=MAX_PER_CLASS):
    out = []
    for split in ['train', 'valid', 'test']:
        df = pd.read_csv(os.path.join(root, 'hc3', f'{domain}_{split}.csv'))
        human = [str(x) for x in df['human_answers'].values.tolist()]
        machine = [str(x) for x in df['chatgpt_answers'].values.tolist()]
        if split == 'train':
            human, machine = _cap(human, max_per_class), _cap(machine, max_per_class)
        out.extend(_pack(human, machine))
    return out


def load_m4(root, domain, model, max_per_class=MAX_PER_CLASS):
    out = []
    for split in ['train', 'valid', 'test']:
        df = pd.read_csv(os.path.join(root, 'm4', f'{domain}_{model}_{split}.csv'))
        human = _clean(df['human_text'].dropna().tolist())
        machine = _clean(df['machine_text'].dropna().tolist())
        if split == 'train':
            human, machine = _cap(human, max_per_class), _cap(machine, max_per_class)
        out.extend(_pack(human, machine))
    return out


def load_raid(root, domain, model, max_per_class=MAX_PER_CLASS):
    out = []
    for split in ['train', 'valid', 'test']:
        dh = pd.read_csv(os.path.join(root, 'raid', f'{domain}_human_{split}.csv'))
        dm = pd.read_csv(os.path.join(root, 'raid', f'{domain}_{model}_{split}.csv'))
        human = _clean(dh['generation'].dropna().tolist())
        machine = _clean(dm['generation'].dropna().tolist())
        n = min(len(human), len(machine))  # balance the two classes
        human, machine = human[:n], machine[:n]
        if split == 'train':
            human, machine = _cap(human, max_per_class), _cap(machine, max_per_class)
        out.extend(_pack(human, machine))
    return out


def _yec_texts(seq):
    """Yelp files are lists of strings; Essay/Creative human files are lists of
    {'prompt': ..., 'essay': ...} dicts whose 'essay' field holds the text."""
    out = []
    for x in seq:
        if isinstance(x, dict):
            x = x.get('essay') or x.get('text') or x.get('generation') or ''
        x = str(x).strip()
        if x:
            out.append(x)
    return out


def _fixed_split(human, machine, max_per_class, fractions=(0.7, 0.1)):
    n = min(len(human), len(machine))
    human, machine = human[:n], machine[:n]
    idx = np.random.RandomState(SPLIT_SEED).permutation(n)
    n_tr, n_va = int(fractions[0] * n), int(fractions[1] * n)
    parts = [idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]]
    out = []
    for k, part in enumerate(parts):
        hh, mm = [human[i] for i in part], [machine[i] for i in part]
        if k == 0:
            hh, mm = _cap(hh, max_per_class), _cap(mm, max_per_class)
        out.extend(_pack(hh, mm))
    return out


def load_yec(root, dataset, model, max_per_class=MAX_PER_CLASS):
    """Yelp / Essay / Creative; no official split, so a fixed 70/10/20 split (seed 42)."""
    folder, tag = YEC_MODELS[model]
    with open(os.path.join(root, folder, f'{dataset}_human.json')) as f:
        human = _yec_texts(json.load(f))
    with open(os.path.join(root, folder, f'{dataset}_{tag}.json')) as f:
        machine = _yec_texts(json.load(f))
    return _fixed_split(human, machine, max_per_class)


def load_pubmed(root, domain, model, max_per_class=MAX_PER_CLASS):
    """PubMed in DetectGPT / DNA-GPT format; 150 pairs -> 75 train / 25 valid / 50 test pairs."""
    with open(os.path.join(root, 'pubmed', f'pubmed_{model}.json'), encoding='utf-8') as f:
        data = json.load(f)
    human, machine = data['original'], data['sampled']
    idx = np.random.RandomState(SPLIT_SEED).permutation(len(human))
    out = []
    for part in [idx[75:], idx[50:75], idx[:50]]:
        out.extend(_pack([human[i] for i in part], [machine[i] for i in part]))
    return out


LOADERS = {'hc3': load_hc3, 'm4': load_m4, 'raid': load_raid,
           'yec': load_yec, 'pubmed': load_pubmed}


def load_splits(bench, root, domain, model=None, max_per_class=MAX_PER_CLASS):
    if bench not in LOADERS:
        raise ValueError(f'unknown benchmark {bench}; choose from {sorted(LOADERS)}')
    return LOADERS[bench](root, domain, model, max_per_class=max_per_class)


def load_cross_splits(bench, root, src_domain, tgt_domain, model=None,
                      max_per_class=MAX_PER_CLASS):
    """Cross-domain: train/valid from the source domain, test from the target domain.
    All graph statistics are then computed on the source training corpus only."""
    src = load_splits(bench, root, src_domain, model, max_per_class)
    tgt = load_splits(bench, root, tgt_domain, model, max_per_class)
    return src[:4] + tgt[4:]
