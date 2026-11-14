"""MGT detection with the default LM2otifs detector (2-layer GCN on the corpus graph).

Covers Table 1 and Tables 7-16 (in-domain), Table 9 (cross-domain, --cross_src),
Table 17 (tokenizer), Table 18 (-U / -W / -Bert), Table 20 (window size) and the
seed statistics of Appendix E.1 (transductive vs. leakage-free, --transductive).

Examples
  # HC3, four domains, leakage-free protocol, 5 seeds
  python scripts/run_detection.py --bench hc3 --seeds 0 1 2 3 4
  # M4, one generator
  python scripts/run_detection.py --bench m4 --domains wikihow reddit peerread arxiv --models davinci
  # ablations on HC3
  python scripts/run_detection.py --unordered            # LM2otifs-U
  python scripts/run_detection.py --weighted             # LM2otifs-W (log(1+x) edge values)
  python scripts/run_detection.py --node_init bert       # BERT token embeddings
  python scripts/run_detection.py --tokenizer gpt2       # Table 17
  python scripts/run_detection.py --window 10            # Table 20
  # cross-domain: train on open_qa, test on the other HC3 domains
  python scripts/run_detection.py --cross_src open_qa --domains wiki_csai medicine finance
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm2otifs.cli import (add_common_args, aggregate_seeds, get_components, graph_tag,
                          iter_combos, load_json, run_and_evaluate, save_json, detector_tag)
from lm2otifs.trainer import disable_jit_fuser


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--cross_src', default=None,
                   help='train on this domain, test on every --domains entry')
    p.add_argument('--no_strict', action='store_true', help='skip the per-document evaluation')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    disable_jit_fuser()

    out = args.out or os.path.join(args.work_dir, f'detection_{args.bench}_{graph_tag(args)}.json')
    results = load_json(out)
    for domain, model in iter_combos(args):
        if args.cross_src and args.cross_src == domain:
            continue
        rows = []
        for seed in args.seeds:
            key = f"{args.cross_src + '->' if args.cross_src else ''}{domain}|{model}|{detector_tag(args, seed)}"
            if key not in results:
                comp, d = get_components(args, domain, model, args.cross_src)
                res, _, _ = run_and_evaluate(args, comp, d, seed, strict=not args.no_strict)
                res.update({'domain': domain, 'model': model, 'seed': seed,
                            'cross_src': args.cross_src,
                            'n_train': comp['n_train'], 'n_val': comp['n_val'],
                            'n_test': comp['n_test'],
                            'vocab': comp['number_nodes'] - comp['num_docs'],
                            'n_word_word_edges': len(comp['ww'][0])})
                results[key] = res
                save_json(results, out)
            rows.append(results[key])
            r = results[key]
            acc = r.get('accuracy_strict', r['accuracy_batched'])
            auc = r.get('auroc_strict', r['auroc_batched'])
            print(f'{key}: ACC {acc:.4f}  AUROC {auc:.4f}')
        if len(rows) > 1:
            agg = aggregate_seeds(rows, ['accuracy_strict', 'auroc_strict',
                                         'accuracy_batched', 'auroc_batched'])
            results[f"{domain}|{model}|summary"] = agg
            save_json(results, out)
            print(f'{domain}|{model} over {len(rows)} seeds:',
                  {k: f"{v['mean']:.4f}±{v['std']:.4f}" for k, v in agg.items()})
    print('saved', out)


if __name__ == '__main__':
    main()
