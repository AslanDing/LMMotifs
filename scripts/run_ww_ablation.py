"""What the co-occurrence structure contributes (Section 5.3, Figure 6).

(1) Training-time ablation: retrain the identical pipeline with the full graph and
    without word-word edges (a bipartite, bag-of-words graph); strict evaluation.
(2) Inference-time ablation (full model): for each test document, delete all
    word-word edges among its own tokens, versus deleting a size-matched random set
    of its document-word edges (all of them when the document has fewer), and
    record the change in the predicted-class probability.

  python scripts/run_ww_ablation.py --seeds 0 1 2
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm2otifs.cli import (add_common_args, aggregate_seeds, get_components, get_detector,
                          graph_tag, iter_combos, load_json, save_json)
from lm2otifs.explain import DocGraph, prob_with_mask
from lm2otifs.graph import EMPTY
from lm2otifs.trainer import disable_jit_fuser, evaluate


def inference_ablation(model, bundle, n_docs, seed):
    rng = np.random.RandomState(seed)
    comp = bundle.comp
    rows = {'p_full': [], 'drop_ww': [], 'drop_dw': [], 'n_ww': [], 'n_dw': []}
    for doc_id in tqdm(comp['test_ids'][:n_docs], desc='inference ablation', leave=False):
        dg = DocGraph(comp, bundle.x, doc_id, bundle.x.device, bundle.edge_norm)
        if len(dg.tokens) < 5 or len(dg.ww_pos) < 10:
            continue
        p = prob_with_mask(model, dg, [])
        pred = int(np.argmax(p))
        k = min(len(dg.ww_pos), len(dg.dw_pos))
        rows['p_full'].append(float(p[pred]))
        rows['drop_ww'].append(float(p[pred] - prob_with_mask(model, dg, dg.ww_pos.tolist())[pred]))
        rows['drop_dw'].append(float(p[pred] - prob_with_mask(
            model, dg, rng.choice(dg.dw_pos, k, replace=False).tolist())[pred]))
        rows['n_ww'].append(len(dg.ww_pos))
        rows['n_dw'].append(len(dg.dw_pos))
    return {'n_docs': len(rows['p_full']),
            'p_full_mean': float(np.mean(rows['p_full'])),
            'prob_change_removing_word_word_edges': float(np.mean(rows['drop_ww'])),
            'prob_change_removing_doc_word_edges': float(np.mean(rows['drop_dw'])),
            'n_word_word_edges_mean': float(np.mean(rows['n_ww'])),
            'n_doc_word_edges_mean': float(np.mean(rows['n_dw']))}


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--inference_docs', type=int, default=200,
                   help='test documents for the inference-time ablation (0 = skip)')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    if args.transductive:
        p.error('the W-W ablation is defined for the leakage-free protocol only')
    disable_jit_fuser()
    out = args.out or os.path.join(args.work_dir, f'ww_ablation_{args.bench}_{graph_tag(args)}.json')
    results = load_json(out)

    for domain, model in iter_combos(args):
        comp, d = get_components(args, domain, model)
        for variant, ww in [('full', None), ('no_ww', EMPTY)]:
            rows = []
            for seed in args.seeds:
                key = f'{domain}|{model}|{variant}|seed{seed}'
                if key not in results:
                    net, b, info = get_detector(args, comp, d, seed, ww=ww,
                                                extra_tag='' if ww is None else '_noww')
                    res = evaluate(net, b)
                    res.update(info)
                    res['n_word_word_edges'] = 0 if ww is not None else len(comp['ww'][0])
                    if variant == 'full' and args.inference_docs:
                        res['inference_ablation'] = inference_ablation(net, b, args.inference_docs, seed)
                    results[key] = res
                    save_json(results, out)
                rows.append(results[key])
                print(f"{key}: ACC {results[key]['accuracy_strict']:.4f} "
                      f"AUROC {results[key]['auroc_strict']:.4f}")
            agg = aggregate_seeds(rows, ['accuracy_strict', 'auroc_strict'])
            results[f'{domain}|{model}|{variant}|summary'] = agg
            save_json(results, out)
            print(f'{domain} {variant}:', {k: f"{v['mean']:.4f}±{v['std']:.4f}" for k, v in agg.items()})
    print('saved', out)


if __name__ == '__main__':
    main()
