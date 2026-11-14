"""Motif faithfulness in the extractor's native, edge-level form
(Figure 5, Figure 8 and its leakage-free re-verification in Appendix E.2;
lambda sensitivity of Figure 7 / Figure 11 / Table 21 via --edge_size).

For each test document, GNNExplainer learns a mask over every edge of the
document's one-hop neighbourhood (word-word edges among its tokens and its
document-word edges). Edges are then deleted most-relevant-first (MoRF) or
least-relevant-first (LeRF) and compared with a random edge ordering.
Reported: probability of the predicted class and accuracy along both curves,
comprehensiveness AOPC and the MoRF-LeRF gap.

  python scripts/run_motif_faithfulness.py --n_docs 60
  python scripts/run_motif_faithfulness.py --edge_size 0.001 0.005 0.01 0.05 0.1 0.5   # lambda sweep
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm2otifs.cli import (add_common_args, get_components, get_detector, graph_tag,
                          iter_combos, load_json, save_json)
from lm2otifs.explain import (BUDGETS, DocGraph, deletion_curves, gnnexplainer_edge_mask,
                              summarize_curves)
from lm2otifs.trainer import disable_jit_fuser


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--n_docs', type=int, default=60, help='test documents to explain')
    p.add_argument('--edge_size', nargs='+', type=float, default=[0.05],
                   help='GNNExplainer size coefficient lambda (several values = sweep)')
    p.add_argument('--edge_ent', type=float, default=0.5)
    p.add_argument('--explain_epochs', type=int, default=150)
    p.add_argument('--explain_lr', type=float, default=0.05)
    p.add_argument('--budgets', nargs='+', type=float, default=BUDGETS)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    if args.transductive:
        p.error('explanations are evaluated on the leakage-free strict per-document graph')
    disable_jit_fuser()
    out = args.out or os.path.join(args.work_dir, f'motif_faithfulness_{args.bench}_{graph_tag(args)}.json')
    results = load_json(out)

    for domain, model in iter_combos(args):
        comp, d = get_components(args, domain, model)
        for seed in args.seeds:
            net, b, _ = get_detector(args, comp, d, seed)
            for lam in args.edge_size:
                key = f'{domain}|{model}|seed{seed}|lambda{lam}'
                if key in results:
                    continue
                rng = np.random.RandomState(seed)
                curves = {'motif': [], 'random': []}
                labels = []
                for doc_id in tqdm(comp['test_ids'][:args.n_docs], desc=key, leave=False):
                    dg = DocGraph(comp, b.x, doc_id, args.device, b.edge_norm)
                    if len(dg.tokens) < 5 or len(dg.ww_pos) < 10:
                        continue
                    edges = np.concatenate([dg.ww_pos, dg.dw_pos]).tolist()
                    imp = gnnexplainer_edge_mask(net, dg, edges, epochs=args.explain_epochs,
                                                 lr=args.explain_lr, edge_size=lam,
                                                 edge_ent=args.edge_ent, seed=seed)
                    for name, scores in [('motif', imp), ('random', rng.rand(len(edges)))]:
                        curves[name].append(deletion_curves(net, dg, edges, scores, list, args.budgets))
                    labels.append(comp['labels'][doc_id])
                res = {'budgets': args.budgets, 'lambda': lam, 'seed': seed}
                for name in curves:
                    res[name] = summarize_curves(curves[name], labels)
                results[key] = res
                save_json(results, out)
                m, r = res['motif'], res['random']
                print(f"{key}: comprehensiveness AOPC motif {m['comprehensiveness_aopc']:.4f} "
                      f"vs random {r['comprehensiveness_aopc']:.4f}; MoRF-LeRF gap "
                      f"{m['morf_lerf_gap_aopc']:.4f} vs {r['morf_lerf_gap_aopc']:.4f}")
    print('saved', out)


if __name__ == '__main__':
    main()
