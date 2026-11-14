"""Shared command-line options, graph caching and detector loading for scripts/."""
import argparse
import json
import os
import re

import numpy as np
import torch

from .data import load_cross_splits, load_splits
from .graph import build_components
from .models import TextGCN
from .trainer import NODE_INITS, evaluate, make_bundle, set_seed, train_detector


def add_common_args(p: argparse.ArgumentParser):
    g = p.add_argument_group('data / graph')
    g.add_argument('--bench', default='hc3', choices=['hc3', 'm4', 'raid', 'yec', 'pubmed'])
    g.add_argument('--domains', nargs='+', default=['open_qa', 'wiki_csai', 'medicine', 'finance'],
                   help='HC3/M4/RAID domains, Yelp/Essay/Creative for yec, ignored for pubmed')
    g.add_argument('--models', nargs='+', default=[None],
                   help='generator: M4 {davinci,cohere,dolly,bloomz,chatgpt}, '
                        'RAID {chatgpt,gpt4,llama-chat,mistral,mpt}, '
                        'yec {sonnet,opus,gemini,gpt35,gpt4}, pubmed {gpt-4,davinci,...}')
    g.add_argument('--data_root', default='data')
    g.add_argument('--work_dir', default='runs', help='graph caches, checkpoints and results')
    g.add_argument('--max_per_class', type=int, default=1000)
    g.add_argument('--tokenizer', default='google-bert/bert-base-uncased')
    g.add_argument('--window', type=int, default=20)
    g.add_argument('--unordered', action='store_true',
                   help='count both orders inside a window ("-U" ablation)')
    g.add_argument('--transductive', action='store_true',
                   help='compute PMI/IDF over train+valid+test (comparison only; NOT the paper protocol)')
    g = p.add_argument_group('detector')
    g.add_argument('--weighted', action='store_true', help='consume edge values ("-W" / "w")')
    g.add_argument('--edge_norm', default=None, choices=['raw', 'log1p'],
                   help='edge-value transform for weighted models (default: log1p if --weighted)')
    g.add_argument('--node_init', default='identity', choices=NODE_INITS)
    g.add_argument('--hidden', type=int, default=64)
    g.add_argument('--epochs', type=int, default=5000)
    g.add_argument('--lr', type=float, default=5e-4)
    g.add_argument('--weight_decay', type=float, default=1e-6)
    g.add_argument('--seeds', nargs='+', type=int, default=[0])
    g.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return p


def edge_norm_of(args):
    if args.edge_norm is not None:
        return args.edge_norm
    return 'log1p' if args.weighted else 'raw'


def _slug(s):
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', str(s))


def graph_tag(args):
    tok = _slug(args.tokenizer.split('/')[-1])
    proto = 'trans' if args.transductive else 'ind'
    order = 'unord' if args.unordered else 'ord'
    return f'{tok}_w{args.window}_{order}_{proto}_cap{args.max_per_class}'


def detector_tag(args, seed, extra=''):
    w = f"w-{edge_norm_of(args)}" if args.weighted else 'uw'
    return f'gcn_{w}_{args.node_init}_h{args.hidden}_e{args.epochs}{extra}_seed{seed}'


def combo_dir(args, domain, model, cross_src=None):
    name = f'{cross_src}__to__{domain}' if cross_src else str(domain)
    if model:
        name += f'_{model}'
    return os.path.join(args.work_dir, args.bench, _slug(name), graph_tag(args))


def get_components(args, domain, model, cross_src=None):
    d = combo_dir(args, domain, model, cross_src)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, 'components.pt')
    if os.path.exists(path):
        return torch.load(path), d
    if cross_src:
        splits = load_cross_splits(args.bench, args.data_root, cross_src, domain, model,
                                   args.max_per_class)
    else:
        splits = load_splits(args.bench, args.data_root, domain, model, args.max_per_class)
    comp = build_components(*splits, tokenizer=args.tokenizer, window_size=args.window,
                            ordered=not args.unordered, transductive=args.transductive)
    if len(comp['ww'][0]) == 0 or len(comp['train_dw'][0]) == 0:
        raise ValueError(f'empty graph for {args.bench}/{domain}/{model}; check the data files')
    torch.save(comp, path)
    return comp, d


def get_detector(args, comp, d, seed, retrain=False, ww=None, extra_tag=''):
    """Load the GCN detector for (graph, seed) from ``d`` or train it."""
    bundle = make_bundle(comp, args.device, args.node_init, edge_norm_of(args), ww=ww)
    set_seed(seed)
    model = TextGCN(bundle.in_dim, args.hidden, 2, weighted=args.weighted).to(args.device)
    ckpt = os.path.join(d, detector_tag(args, seed, extra_tag) + '.pt')
    if os.path.exists(ckpt) and not retrain:
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
        model.eval()
        info = {'loaded': ckpt}
    else:
        info = train_detector(model, bundle, ckpt, args.epochs, args.lr, args.weight_decay)
    return model, bundle, info


def run_and_evaluate(args, comp, d, seed, strict=True, **kw):
    model, bundle, info = get_detector(args, comp, d, seed, **kw)
    res = evaluate(model, bundle, strict=strict)
    res.update(info)
    return res, model, bundle


def aggregate_seeds(rows, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}
    return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def iter_combos(args):
    domains = args.domains if args.bench != 'pubmed' else ['pubmed']
    for domain in domains:
        for model in args.models:
            yield domain, model
