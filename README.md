# LM²otifs

Code for **"Towards Structurally Graph-Based Explainable Machine-Generated Text Detection"**.

LM²otifs moves machine-generated text (MGT) detection from linear token sequences to a
corpus-level lexical co-occurrence graph. Word–word edges carry positive PMI over sliding
windows, document–word edges carry TF-IDF, and a GNN classifies the document nodes. Because
word pairs are first-class objects in the graph, a graph explainer can return
**motifs**: co-occurrence structures that separate human from machine text.

This repository contains the code for the key experiments of the paper: graph construction
under the leakage-free protocol, the detector and its ablations, the architecture sweep, and
the explanation experiments (motif faithfulness, the unified token-level protocol, the canary
with planted ground truth, and corpus-level motif statistics).

## Layout

```
lm2otifs/
  data.py      dataset loaders (HC3, M4, RAID, Yelp/Essay/Creative, PubMed)
  graph.py     corpus graph: PMI word-word edges, TF-IDF doc-word edges, leakage-free views
  models.py    TextGCN (default detector) and SweepGNN (GCN/GAT/SAGE/Transformer/PNA)
  trainer.py   node features, training loop, batched and strict per-document evaluation
  explain.py   GNNExplainer edge masks, corpus-level motif mask, token attributors, MoRF/LeRF
  cli.py       shared command-line options, graph/checkpoint caching
scripts/
  preprocess_hc3.py            build the HC3 splits
  run_detection.py             detection, ablations, cross-domain, seeds
  run_arch_sweep.py            five architectures, unweighted / weighted
  run_ww_ablation.py           full graph vs. no word-word edges; inference-time deletion
  run_motif_faithfulness.py    edge-level MoRF/LeRF of motifs vs. random; lambda sweep
  run_unified_attribution.py   LIME / KernelSHAP / gradient x edge / random on the same GCN
  run_canary.py                planted-evidence canary (cooc and bag regimes)
  run_global_motifs.py         corpus-level motifs and HGT/MGT coverage statistics
```

## Installation

```bash
conda create -n lm2otifs python=3.9 -y && conda activate lm2otifs
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.9 torch-sparse==0.6.13 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install -r requirements.txt
```

Tested with Python 3.9, PyTorch 1.10.0 (CUDA 11.3) and PyTorch Geometric 2.0.4.

## Data

All scripts read from `--data_root` (default `data/`):

| Benchmark | Source | Expected files |
|---|---|---|
| HC3 | [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) `all.jsonl` | `hc3/{domain}_{train,valid,test}.csv`, built by `python scripts/preprocess_hc3.py --raw <all.jsonl>` |
| M4 | [M4](https://github.com/mbzuai-nlp/M4) | `m4/{domain}_{model}_{train,valid,test}.csv` with columns `human_text`, `machine_text` |
| RAID | [RAID](https://github.com/liamdugan/raid) | `raid/{domain}_human_{split}.csv`, `raid/{domain}_{model}_{split}.csv` with column `generation` |
| Yelp / Essay / Creative | [BiScope release](https://github.com/MarkGHX/BiScope) (Yelp from Raidar, Essay/Creative from Ghostbuster) | `{claude3-sonnet,claude3-opus,Gemini,GPT-3.5,GPT-4}/{Dataset}_human.json` and `{Dataset}_{model-tag}.json` |
| PubMed | [DNA-GPT](https://github.com/Xianjun-Yang/DNA-GPT) | `pubmed/pubmed_{model}.json` with keys `original`, `sampled` |

HC3 domains: `open_qa wiki_csai medicine finance`; M4 domains: `wikihow reddit peerread arxiv`
with generators `davinci cohere dolly bloomz chatgpt`; RAID domains: `recipes books poetry reviews`
with generators `chatgpt gpt4 llama-chat mistral mpt`; Yelp/Essay/Creative generators
`sonnet opus gemini gpt35 gpt4`. Training sets are capped at 1000 documents per class.

## Reproducing the experiments

Graphs, checkpoints and JSON results are cached under `--work_dir` (default `runs/`).
A detector trained by one script is reused by the others when the graph and detector options
match. Every command below accepts `--bench`, `--domains`, `--models`, `--seeds` and `--device`.

| Paper | Command |
|---|---|
| Table 1, Tables 7–8 (HGT vs. ChatGPT) | `python scripts/run_detection.py --bench hc3` <br> `python scripts/run_detection.py --bench m4 --domains wikihow reddit peerread arxiv --models chatgpt` <br> `python scripts/run_detection.py --bench raid --domains recipes books poetry reviews --models chatgpt` |
| Tables 10–16 (other generators) | e.g. `python scripts/run_detection.py --bench raid --domains recipes poetry reviews --models gpt4 llama-chat mistral mpt` <br> `python scripts/run_detection.py --bench yec --domains Yelp Essay Creative --models sonnet opus gemini` |
| Table 9 (cross-domain) | `python scripts/run_detection.py --bench hc3 --cross_src open_qa --domains wiki_csai medicine finance` |
| Table 2 (architectures) | `python scripts/run_arch_sweep.py --seeds 0 1 2 3 4` <br> `python scripts/run_arch_sweep.py --archs gcn gat transformer pna --weighted --seeds 0 1 2 3 4` |
| Seed statistics, App. E.1 | `python scripts/run_detection.py --seeds 0 1 2 3 4` (add `--transductive` for the transductive comparison) |
| Table 18 (ablations) | `--unordered` (-U), `--weighted` (-W), `--unordered --weighted` (-UW), `--node_init bert` (-Bert) |
| Table 17 (tokenizer) | `python scripts/run_detection.py --tokenizer gpt2` |
| Table 20 (window size) | `python scripts/run_detection.py --window 10` (15, 20, 25, 30) |
| Figure 6 (word–word structure) | `python scripts/run_ww_ablation.py --domains open_qa wiki_csai medicine finance --seeds 0 1 2` |
| Figures 5 and 8, App. E.2 re-verification (motifs vs. random, edge level) | `python scripts/run_motif_faithfulness.py --n_docs 60` |
| Figure 7, Figure 11, Table 21 (lambda) | `python scripts/run_motif_faithfulness.py --edge_size 0.001 0.005 0.01 0.05 0.1 0.5` |
| Figures 9 and 10 (unified token-level protocol) | `python scripts/run_unified_attribution.py --methods lime shap gradient random` |
| Table 22 (canary) | `python scripts/run_canary.py --mode cooc` and `python scripts/run_canary.py --mode bag` |
| Tables 23–25 (motif statistics), Tables 26–27 (motif samples) | `python scripts/run_global_motifs.py` (use `--bench pubmed --models gpt-4 claude-3 davinci` for the PubMed samples) |

## Protocol and implementation notes

* **Leakage-free graph.** Vocabulary, PMI window statistics and IDF come from the training
  documents only and are then frozen. Out-of-vocabulary tokens of validation and test documents
  are dropped. During training, validation and test document nodes are isolated. Validation
  edges are added for model selection.
* **Strict per-document evaluation.** Each test document is inserted on its own into the frozen
  train+validation graph (`evaluate_strict`). We also report the batched setting, in which all
  test documents are inserted at once. `--transductive` rebuilds the original TextGCN-style graph,
  with statistics over train+valid+test, and is provided only for comparison.
* **Detector.** Two GCN layers (|V| → 64 → 2, SiLU, dropout before the second layer), AdamW with
  learning rate 5e-4 and weight decay 1e-6, 5000 full-batch epochs. We keep the checkpoint with the
  best validation accuracy. As in the reported runs, the validation forward pass runs in train
  mode. All test-time evaluation runs in eval mode.
* **Edge weights.** By default the network sees the binarized adjacency. `--weighted` feeds PMI
  and TF-IDF values, normalized with log(1+x).
* **Node features.** The default `--node_init identity` reproduces the input used for the reported
  runs: a sparse identity block on the first |V| node indices. Because document nodes come
  first, every document node and most token nodes get a one-hot vector.
  `--node_init onehot` gives token nodes one-hot vectors and document nodes zero vectors.
  `--node_init bert` uses BERT input embeddings for token nodes.
* **Explainers.** Motifs are extracted with a GNNExplainer objective: prediction preservation,
  plus lambda times the mask size, plus mask entropy. The mask is applied as edge weights to the
  strict per-document graph, so a zero weight deletes the edge. Defaults: lambda = 0.05, entropy
  0.5, Adam with learning rate 0.05, 150 steps. Corpus-level motifs use one mask over all
  word–word edges that preserves predictions on the training documents.
* **PNA** receives the in-degree histogram of the training graph, as PyTorch Geometric expects.

Not included: the external detection baselines (use their official implementations, e.g.
[Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt), which provides the zero-shot
baselines with OPT-2.7B as reference model) and the system-level comparison that explains
RoBERTa-QA with LIME, SHAP, GLTR and GPT-4o.
