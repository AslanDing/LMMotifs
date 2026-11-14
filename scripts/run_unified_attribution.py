"""Unified token-level protocol (Appendix E.2, Figures 9 and 10).

Every method explains the SAME trained GCN, every output is a ranking over the
test document's in-vocabulary tokens, and the deletion operator is shared:
deleting a token removes its document-word edges from the strict per-document
graph. Methods: LIME, KernelSHAP, gradient x edge, random (and optionally a
GNNExplainer mask scoped to the document's doc-word edges).

  python scripts/run_unified_attribution.py --methods lime shap gradient random
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm2otifs.cli import (add_common_args, get_components, get_detector, graph_tag,
                          iter_combos, load_json, save_json)
from lm2otifs.explain import (BUDGETS, TOKEN_ATTRIBUTORS, DocGraph, deletion_curves,
                              summarize_curves)
from lm2otifs.trainer import disable_jit_fuser


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--methods', nargs='+', default=['lime', 'shap', 'gradient', 'random'],
                   choices=sorted(TOKEN_ATTRIBUTORS))
    p.add_argument('--n_docs', type=int, default=None, help='default: all test documents')
    p.add_argument('--budgets', nargs='+', type=float, default=BUDGETS)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    if args.transductive:
        p.error('explanations are evaluated on the leakage-free strict per-document graph')
    disable_jit_fuser()
    out = args.out or os.path.join(args.work_dir, f'unified_attribution_{args.bench}_{graph_tag(args)}.json')
    results = load_json(out)

    for domain, model in iter_combos(args):
        comp, d = get_components(args, domain, model)
        for seed in args.seeds:
            key = f'{domain}|{model}|seed{seed}'
            if key in results and all(m in results[key]['methods'] for m in args.methods):
                continue
            net, b, _ = get_detector(args, comp, d, seed)
            rng = np.random.RandomState(seed)
            per_method = {m: [] for m in args.methods}
            labels = []
            for doc_id in tqdm(comp['test_ids'][:args.n_docs], desc=key, leave=False):
                dg = DocGraph(comp, b.x, doc_id, args.device, b.edge_norm)
                if len(dg.tokens) < 5:
                    continue
                labels.append(comp['labels'][doc_id])
                for m in args.methods:
                    scores = TOKEN_ATTRIBUTORS[m](model=net, dg=dg, rng=rng, seed=seed)
                    per_method[m].append(deletion_curves(net, dg, dg.tokens, scores,
                                                         dg.positions, args.budgets))
            res = {'budgets': args.budgets, 'seed': seed, 'methods': {}}
            for m in args.methods:
                res['methods'][m] = summarize_curves(per_method[m], labels)
                s = res['methods'][m]
                print(f"{key} {m:>12}: comprehensiveness AOPC {s['comprehensiveness_aopc']:.3f}  "
                      f"MoRF-LeRF gap {s['morf_lerf_gap_aopc']:.3f}")
            results[key] = res
            save_json(results, out)
    print('saved', out)


if __name__ == '__main__':
    main()
