"""Canary evaluation with planted ground truth (Appendix E.3, Table 22).

All text is human-written (HC3 human answers of the four domains, >= 40 words,
pooled and re-split 70/10/20); class labels are created only by injecting pairs of
signature tokens (single-piece alphabetic BERT-vocabulary words of length 6-12 that
occur nowhere in the corpus), three injections per document.

  cooc : both classes receive the same pair; class 1 places the two tokens adjacently,
         class 0 at least 30 words apart. The token multiset is identical across
         classes (the construction of Proposition 3.1), so by Proposition 3.3 the
         corpus-level detector should stay at chance.
  bag  : the classes receive disjoint pairs (positive control). The planted tokens are
         then a valid ground truth, and each token attributor is scored by whether both
         planted tokens appear in its top-k (hit@k) and by MRR, on up to 60 class-1 test
         documents, only if detection accuracy exceeds 0.7.

  python scripts/run_canary.py --mode cooc
  python scripts/run_canary.py --mode bag
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm2otifs.cli import add_common_args, get_detector, graph_tag, save_json
from lm2otifs.data import HC3_DOMAINS
from lm2otifs.explain import TOKEN_ATTRIBUTORS, DocGraph
from lm2otifs.graph import build_components, get_tokenizer
from lm2otifs.trainer import disable_jit_fuser, evaluate

SEED, MIN_WORDS, GAP, N_PAIRS, REPEATS = 42, 40, 30, 10, 3


def pick_signature_tokens(tokenizer, texts, n):
    seen = set()
    for t in texts:
        seen.update(tokenizer(t)['input_ids'])
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    cands = [w for w, i in vocab if i not in seen and re.fullmatch(r'[a-z]{6,12}', w)]
    np.random.RandomState(SEED).shuffle(cands)
    return cands[:n]


def inject_once(words, w1, w2, adjacent, rng):
    n = len(words)
    if adjacent:
        p = rng.randint(1, n - 1)
        return words[:p] + [w1, w2] + words[p:]
    p1 = rng.randint(1, max(2, n - GAP - 2))
    p2 = rng.randint(p1 + GAP, n)
    return words[:p1] + [w1] + words[p1:p2] + [w2] + words[p2:]


def build_canary_splits(data_root, mode, tokenizer):
    rng = np.random.RandomState(SEED)
    pool = []
    for dom in HC3_DOMAINS:
        for split in ['train', 'valid', 'test']:
            df = pd.read_csv(os.path.join(data_root, 'hc3', f'{dom}_{split}.csv'))
            pool.extend(l for l in df['human_answers'].values.tolist()
                        if isinstance(l, str) and len(l.split()) >= MIN_WORDS)
    rng.shuffle(pool)
    n = len(pool)
    parts = [pool[:int(0.7 * n)], pool[int(0.7 * n):int(0.8 * n)], pool[int(0.8 * n):]]
    sig = pick_signature_tokens(tokenizer, pool, 4 * N_PAIRS)
    pos_pairs = [(sig[2 * i], sig[2 * i + 1]) for i in range(N_PAIRS)]
    neg_pairs = [(sig[2 * N_PAIRS + 2 * i], sig[2 * N_PAIRS + 2 * i + 1]) for i in range(N_PAIRS)]

    def make_split(lines):
        rng.shuffle(lines)
        half = len(lines) // 2
        docs, labels, meta = [], [], []
        for j, line in enumerate(lines[:2 * half]):
            cls = 0 if j < half else 1
            if mode == 'cooc':
                w1, w2 = pos_pairs[j % N_PAIRS]
                adjacent = cls == 1
            else:
                w1, w2 = (pos_pairs if cls == 1 else neg_pairs)[j % N_PAIRS]
                adjacent = True
            words = line.split()
            for _ in range(REPEATS):
                words = inject_once(words, w1, w2, adjacent, rng)
            docs.append(' '.join(words))
            labels.append(cls)
            meta.append({'pair': (w1, w2), 'cls': cls})
        return docs, labels, meta

    tr, va, te = (make_split(part) for part in parts)
    return (tr[0], tr[1], va[0], va[1], te[0], te[1]), pos_pairs, te[2]


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.set_defaults(epochs=3000)
    p.add_argument('--mode', default='cooc', choices=['cooc', 'bag'])
    p.add_argument('--methods', nargs='+', default=['gradient', 'lime', 'shap', 'random'],
                   choices=sorted(TOKEN_ATTRIBUTORS))
    p.add_argument('--n_explain', type=int, default=60)
    p.add_argument('--min_accuracy', type=float, default=0.7)
    p.add_argument('--strict', action='store_true',
                   help='also run the (slow) strict per-document evaluation; detection is gated on batched accuracy')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    disable_jit_fuser()
    seed = args.seeds[0]

    d = os.path.join(args.work_dir, 'canary', args.mode, graph_tag(args))
    os.makedirs(d, exist_ok=True)
    cpath = os.path.join(d, 'components.pt')
    tok = get_tokenizer(args.tokenizer)
    if os.path.exists(cpath):
        comp, pairs, te_meta = torch.load(cpath)
    else:
        splits, pairs, te_meta = build_canary_splits(args.data_root, args.mode, tok)
        comp = build_components(*splits, tokenizer=args.tokenizer, window_size=args.window,
                                ordered=not args.unordered)
        torch.save((comp, pairs, te_meta), cpath)
    print('signature pairs:', pairs)

    net, b, info = get_detector(args, comp, d, seed)
    det = evaluate(net, b, strict=args.strict)
    print(f"canary/{args.mode}: test ACC {det['accuracy_batched']:.3f} AUROC {det['auroc_batched']:.3f}"
          + (f" (strict ACC {det['accuracy_strict']:.3f})" if args.strict else ''))

    word_node = {t: i for i, t in enumerate(comp['vocab'])}
    tok_id = tok.get_vocab()
    hits = {m: {'hit@5': [], 'hit@10': [], 'mrr': []} for m in args.methods}
    cand = [(i, m) for i, m in zip(comp['test_ids'], te_meta) if m['cls'] == 1][:args.n_explain]
    if det['accuracy_batched'] < args.min_accuracy:
        print(f'detection below {args.min_accuracy}: planted tokens are not expressed by the model, '
              'skipping attribution recovery')
        cand = []
    rng = np.random.RandomState(seed)
    for doc_id, meta in tqdm(cand, desc='explain'):
        dg = DocGraph(comp, b.x, doc_id, args.device, b.edge_norm)
        gt = {word_node.get(tok_id[w]) for w in meta['pair']} - {None}
        if not gt or not gt & set(dg.tokens):
            continue
        for m in args.methods:
            scores = TOKEN_ATTRIBUTORS[m](model=net, dg=dg, rng=rng, seed=seed)
            order = list(np.asarray(dg.tokens)[np.argsort(-np.asarray(scores))])
            ranks = sorted(order.index(w) + 1 for w in gt if w in order)
            hits[m]['hit@5'].append(int(all(r <= 5 for r in ranks)))
            hits[m]['hit@10'].append(int(all(r <= 10 for r in ranks)))
            hits[m]['mrr'].append(1.0 / ranks[0])

    summary = {'mode': args.mode, 'seed': seed, 'pairs': [list(x) for x in pairs],
               'detection': det, 'train': info,
               'n_explained_docs': len(next(iter(hits.values()))['mrr']),
               'recovery': {m: {k: float(np.mean(v)) if v else None for k, v in h.items()}
                            for m, h in hits.items()}}
    out = args.out or os.path.join(args.work_dir, f'canary_{args.mode}.json')
    save_json(summary, out)
    for m, r in summary['recovery'].items():
        print(f'{m:>12}: {r}')
    print('saved', out)


if __name__ == '__main__':
    main()
