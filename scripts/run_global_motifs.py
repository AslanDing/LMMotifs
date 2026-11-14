"""Corpus-level motifs and their statistics (Appendix E.3 "Motifs Statistical
Analysis", Tables 23-25; motif samples as in Tables 26-27).

A single GNNExplainer-style mask over all word-word edges is optimised to preserve
the detector's predictions on the training documents of one class. The top
``--edge_sparsity`` fraction of word-word pairs (0.05% in the paper) are the motif
edges and the top ``--node_sparsity`` fraction of tokens (by mean incident mask
weight) the motif nodes. For each, we count how many human (HGT) and machine (MGT)
training documents contain it (an edge counts when both tokens occur in the
document). A random ordering of the same edges is reported as a control.

Outputs <work_dir>/global_motifs_<bench>/<domain>/class{c}_{edges,nodes}.csv and a
JSON summary with the HGT/MGT totals and the Nodes/Edges ratio.

  python scripts/run_global_motifs.py --domains open_qa wiki_csai medicine finance
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm2otifs.cli import (add_common_args, get_components, get_detector, graph_tag,
                          iter_combos, load_json, save_json)
from lm2otifs.explain import corpus_motif_mask
from lm2otifs.graph import get_tokenizer
from lm2otifs.trainer import disable_jit_fuser


def doc_token_sets(comp):
    sets = defaultdict(set)
    rows, cols, _ = comp['train_dw']
    for r, c in zip(rows, cols):
        sets[r].add(c - comp['num_docs'])
    return sets


def motif_statistics(comp, scores, edge_sparsity, node_sparsity):
    num_docs = comp['num_docs']
    rows, cols, _ = comp['ww']
    pair_score = {}
    for i, j, s in zip(rows, cols, scores):
        key = (min(i, j) - num_docs, max(i, j) - num_docs)
        pair_score[key] = max(pair_score.get(key, 0.0), float(s))
    node_sum, node_cnt = defaultdict(float), defaultdict(int)
    for (a, b), s in pair_score.items():
        for w in (a, b):
            node_sum[w] += s
            node_cnt[w] += 1
    node_score = {w: node_sum[w] / node_cnt[w] for w in node_sum}

    top_pairs = sorted(pair_score.items(), key=lambda kv: -kv[1])[:max(1, int(edge_sparsity * len(pair_score)))]
    top_nodes = sorted(node_score.items(), key=lambda kv: -kv[1])[:max(1, int(node_sparsity * len(node_score)))]

    labels = comp['labels']
    sets = doc_token_sets(comp)
    edge_rows, node_rows = [], []
    for (a, b), s in top_pairs:
        cnt = [0, 0]
        for doc, toks in sets.items():
            if a in toks and b in toks:
                cnt[labels[doc]] += 1
        edge_rows.append((a, b, s, cnt[0], cnt[1]))
    for w, s in top_nodes:
        cnt = [0, 0]
        for doc, toks in sets.items():
            if w in toks:
                cnt[labels[doc]] += 1
        node_rows.append((w, s, cnt[0], cnt[1]))
    tot = {'nodes': {'hgt': sum(r[2] for r in node_rows), 'mgt': sum(r[3] for r in node_rows)},
           'edges': {'hgt': sum(r[3] for r in edge_rows), 'mgt': sum(r[4] for r in edge_rows)}}
    tot['nodes_per_edge'] = {k: tot['nodes'][k] / max(1, tot['edges'][k]) for k in ['hgt', 'mgt']}
    tot['n_motif_edges'], tot['n_motif_nodes'] = len(edge_rows), len(node_rows)
    return tot, edge_rows, node_rows


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--classes', nargs='+', type=int, default=[0, 1],
                   help='explain the training documents of these classes (0 = human, 1 = machine)')
    p.add_argument('--edge_sparsity', type=float, default=0.0005)
    p.add_argument('--node_sparsity', type=float, default=0.1)
    p.add_argument('--motif_epochs', type=int, default=300)
    p.add_argument('--motif_lr', type=float, default=0.05)
    p.add_argument('--edge_size', type=float, default=1e-4)
    p.add_argument('--edge_ent', type=float, default=0.1)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    if args.transductive:
        p.error('use the leakage-free graph')
    disable_jit_fuser()
    seed = args.seeds[0]
    tok = get_tokenizer(args.tokenizer)
    out = args.out or os.path.join(args.work_dir, f'global_motifs_{args.bench}_{graph_tag(args)}.json')
    results = load_json(out)

    for domain, model in iter_combos(args):
        comp, d = get_components(args, domain, model)
        net, b, _ = get_detector(args, comp, d, seed)
        csv_dir = os.path.join(args.work_dir, f'global_motifs_{args.bench}', f'{domain}_{model}' if model else domain)
        os.makedirs(csv_dir, exist_ok=True)
        rng = np.random.RandomState(seed)
        for c in args.classes:
            doc_ids = [i for i in range(comp['n_train']) if comp['labels'][i] == c]
            scores = corpus_motif_mask(net, comp, b.x, args.device, doc_ids, epochs=args.motif_epochs,
                                       lr=args.motif_lr, edge_size=args.edge_size,
                                       edge_ent=args.edge_ent, seed=seed, edge_norm=b.edge_norm)
            entry = {}
            for name, sc in [('motif', scores), ('random', rng.permutation(scores))]:
                tot, edge_rows, node_rows = motif_statistics(comp, sc, args.edge_sparsity, args.node_sparsity)
                entry[name] = tot
                if name == 'motif':
                    decode = lambda w: tok.convert_ids_to_tokens(int(comp['vocab'][w]))
                    with open(os.path.join(csv_dir, f'class{c}_edges.csv'), 'w', newline='', encoding='utf-8') as f:
                        wr = csv.writer(f)
                        wr.writerow(['token_i', 'token_j', 'mask_score', 'n_hgt_docs', 'n_mgt_docs'])
                        for a, bb, s, h, m in edge_rows:
                            wr.writerow([decode(a), decode(bb), f'{s:.6f}', h, m])
                    with open(os.path.join(csv_dir, f'class{c}_nodes.csv'), 'w', newline='', encoding='utf-8') as f:
                        wr = csv.writer(f)
                        wr.writerow(['token', 'mean_mask_score', 'n_hgt_docs', 'n_mgt_docs'])
                        for w, s, h, m in node_rows:
                            wr.writerow([decode(w), f'{s:.6f}', h, m])
            results[f'{domain}|{model}|class{c}'] = entry
            save_json(results, out)
            print(f"{domain} class {c}: motif nodes HGT/MGT {entry['motif']['nodes']}, "
                  f"edges {entry['motif']['edges']}, nodes/edges {entry['motif']['nodes_per_edge']}")
    print('saved', out)


if __name__ == '__main__':
    main()
