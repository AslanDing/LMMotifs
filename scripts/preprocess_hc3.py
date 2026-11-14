"""Build the HC3 splits used in the paper from the raw HC3 release.

Download ``all.jsonl`` from https://huggingface.co/datasets/Hello-SimpleAI/HC3 and run

  python scripts/preprocess_hc3.py --raw data/hc3_raw/all.jsonl --out data/hc3

Each domain is shuffled with seed 42; the first 100 questions form the test set,
the next 50 the validation set and the rest the training set. Each question
contributes one human and one ChatGPT answer.
"""
import argparse
import json
import os

import pandas as pd

DOMAINS = ['open_qa', 'wiki_csai', 'medicine', 'finance']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default='data/hc3_raw/all.jsonl')
    p.add_argument('--out', default='data/hc3')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.raw, encoding='utf-8') as f:
        df = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    for name in DOMAINS:
        dom = df[df['source'] == name].sample(frac=1, random_state=args.seed)
        dom.iloc[150:].to_csv(os.path.join(args.out, f'{name}_train.csv'), index=False)
        dom.iloc[100:150].to_csv(os.path.join(args.out, f'{name}_valid.csv'), index=False)
        dom.iloc[:100].to_csv(os.path.join(args.out, f'{name}_test.csv'), index=False)
        print(f'{name}: train {len(dom) - 150}, valid 50, test 100 questions')


if __name__ == '__main__':
    main()
