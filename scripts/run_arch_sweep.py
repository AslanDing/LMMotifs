"""Generality of the detector component (Table 2).

The same leakage-free graph is consumed by five message-passing architectures
(sparse input projection + 2 layers, shared hyper-parameters), unweighted ("uw")
or with log(1+x)-normalised edge values ("w"; SAGE has no edge-weight support).

  python scripts/run_arch_sweep.py --archs gcn gat sage transformer pna --seeds 0 1 2 3 4
  python scripts/run_arch_sweep.py --archs gcn gat transformer pna --weighted --seeds 0 1 2 3 4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm2otifs.cli import (add_common_args, aggregate_seeds, edge_norm_of, get_components,
                          graph_tag, iter_combos, load_json, save_json)
from lm2otifs.models import ARCHS, WEIGHTED_SUPPORT, SweepGNN, pna_degree_histogram
from lm2otifs.trainer import disable_jit_fuser, evaluate, make_bundle, set_seed, train_detector


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--archs', nargs='+', default=ARCHS, choices=ARCHS)
    p.add_argument('--no_strict', action='store_true')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    disable_jit_fuser()
    norm = edge_norm_of(args)
    out = args.out or os.path.join(args.work_dir, f'arch_sweep_{args.bench}_{graph_tag(args)}.json')
    results = load_json(out)

    for arch in args.archs:
        if args.weighted and arch not in WEIGHTED_SUPPORT:
            print(f'skip {arch}: no edge-weight support')
            continue
        tag = f"{arch}_{'w-' + norm if args.weighted else 'uw'}"
        for domain, model in iter_combos(args):
            rows = []
            for seed in args.seeds:
                key = f'{tag}|{domain}|{model}|seed{seed}'
                if key not in results:
                    comp, d = get_components(args, domain, model)
                    b = make_bundle(comp, args.device, args.node_init, norm)
                    deg = pna_degree_histogram(b.g_train.edge_index, comp['number_nodes']) \
                        if arch == 'pna' else None
                    set_seed(seed)
                    net = SweepGNN(arch, b.in_dim, args.hidden, 2, args.weighted, deg=deg).to(args.device)
                    info = train_detector(net, b, os.path.join(d, f'sweep_{tag}_e{args.epochs}_seed{seed}.pt'),
                                          args.epochs, args.lr, args.weight_decay)
                    res = evaluate(net, b, strict=not args.no_strict)
                    res.update(info)
                    res.update({'arch': arch, 'weighted': args.weighted, 'edge_norm': norm,
                                'domain': domain, 'model': model, 'seed': seed})
                    results[key] = res
                    save_json(results, out)
                rows.append(results[key])
                r = results[key]
                print(f"{key}: ACC {r.get('accuracy_strict', r['accuracy_batched']):.4f}")
            if len(rows) > 1:
                agg = aggregate_seeds(rows, ['accuracy_strict', 'accuracy_batched', 'auroc_strict'])
                results[f'{tag}|{domain}|{model}|summary'] = agg
                save_json(results, out)
                print(f'{tag}|{domain}:', {k: f"{v['mean']:.3f}±{v['std']:.3f}" for k, v in agg.items()})
    print('saved', out)


if __name__ == '__main__':
    main()
