"""Training and evaluation of document-node classifiers on the corpus graph."""
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm

from .graph import graph_views, single_doc_adj, to_pyg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def disable_jit_fuser():
    """torch 1.10's TensorExpr fuser does not support recent GPUs (e.g. sm_89);
    attention-style convolutions crash without this. Harmless elsewhere."""
    try:
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
    except Exception:
        pass


# ---------------------------------------------------------------- node features

def identity_features(comp, device):
    """Sparse [num_nodes, |V|] matrix with X[r, r] = 1 for the first |V| node indices.

    This is the input used for every run reported in the paper (the original
    ``one_hot_node_init``). Because document nodes come first in the index space,
    every document node and the first |V| - num_docs token nodes receive an identity
    feature, and the last num_docs token nodes receive a zero vector.
    """
    n_words = comp['number_nodes'] - comp['num_docs']
    idx = torch.arange(n_words)
    return torch.sparse_coo_tensor(torch.stack([idx, idx]), torch.ones(n_words),
                                   size=(comp['number_nodes'], n_words)).to(device)


def one_hot_features(comp, device):
    """Token nodes get one-hot features and document nodes zero vectors (Sec. 4.1 wording)."""
    n_words = comp['number_nodes'] - comp['num_docs']
    idx = torch.arange(n_words)
    ind = torch.stack([idx + comp['num_docs'], idx])
    return torch.sparse_coo_tensor(ind, torch.ones(n_words),
                                   size=(comp['number_nodes'], n_words)).to(device)


def bert_features(comp, device, model_name='google-bert/bert-base-uncased'):
    """Token nodes initialised with BERT input embeddings (the "-Bert" row of Table 18)."""
    try:
        from transformers import AutoModel
        emb = AutoModel.from_pretrained(model_name).get_input_embeddings().weight.detach()
    except Exception:  # e.g. recent transformers refuse torch<2.1: read the checkpoint directly
        from huggingface_hub import hf_hub_download
        state = torch.load(hf_hub_download(model_name, 'pytorch_model.bin'), map_location='cpu')
        emb = state[[k for k in state if k.endswith('word_embeddings.weight')][0]]
    x = torch.zeros(comp['number_nodes'], emb.shape[1])
    x[comp['num_docs']:] = emb[torch.as_tensor(comp['vocab'])]
    return x.to(device)


NODE_INITS = ['identity', 'onehot', 'bert']


def node_features(comp, device, node_init='identity'):
    if node_init == 'identity':
        return identity_features(comp, device)
    if node_init == 'onehot':
        return one_hot_features(comp, device)
    if node_init == 'bert':
        return bert_features(comp, device)
    raise ValueError(node_init)


# ---------------------------------------------------------------- graph bundle

@dataclass
class Bundle:
    comp: dict
    x: torch.Tensor
    g_train: object
    g_val: object
    g_full: object
    labels: torch.Tensor
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    test_ids: torch.Tensor
    edge_norm: str = 'raw'
    ww: tuple = None          # word-word edges actually used (None = all)

    @property
    def in_dim(self):
        return self.x.shape[1]


def make_bundle(comp, device, node_init='identity', edge_norm='raw', ww=None):
    x = node_features(comp, device, node_init)
    adj_train, adj_val, adj_full = graph_views(comp, ww)
    split = comp['split']
    ids = {s: torch.tensor([i for i, v in enumerate(split) if v == s], device=device)
           for s in ['train', 'val', 'test']}
    return Bundle(comp=comp, x=x,
                  g_train=to_pyg(adj_train, x, device, edge_norm),
                  g_val=to_pyg(adj_val, x, device, edge_norm),
                  g_full=to_pyg(adj_full, x, device, edge_norm),
                  labels=torch.tensor(comp['labels'], device=device),
                  train_ids=ids['train'], val_ids=ids['val'], test_ids=ids['test'],
                  edge_norm=edge_norm, ww=ww)


# ---------------------------------------------------------------- metrics

def classification_metrics(logits, y_true):
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true)
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob = prob / prob.sum(1, keepdims=True)
    y_pred = logits.argmax(1)
    out = {'accuracy': float(metrics.accuracy_score(y_true, y_pred)),
           'f1_macro': float(metrics.f1_score(y_true, y_pred, average='macro'))}
    try:
        out['auroc'] = float(metrics.roc_auc_score(y_true, prob[:, 1]))
    except ValueError:
        out['auroc'] = float('nan')
    return out


# ---------------------------------------------------------------- training

def train_detector(model, b: Bundle, ckpt, epochs=5000, lr=5e-4, weight_decay=1e-6,
                   log_every=0):
    """Full-batch training on the training graph; the checkpoint with the best
    validation accuracy (ties -> later epoch) is kept.

    Note: as in the runs reported in the paper, the validation forward pass is made
    with the network still in train mode (dropout active). Test-time evaluation
    (``evaluate_*``) always runs in eval mode.
    """
    os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    y = b.labels
    best_val, best_epoch = -1.0, -1
    for epoch in tqdm(range(epochs), desc=os.path.basename(ckpt), leave=False):
        model.train()
        opt.zero_grad()
        logits = model(b.g_train)
        loss = F.cross_entropy(logits[b.train_ids], y[b.train_ids])
        loss.backward()
        opt.step()
        with torch.no_grad():
            val_pred = model(b.g_val)[b.val_ids].argmax(-1)
            val_acc = (val_pred == y[b.val_ids]).float().mean().item()
        if best_val <= val_acc:
            best_val, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), ckpt)
        if log_every and epoch % log_every == 0:
            print(f'epoch {epoch:5d} loss {loss.item():.4f} val_acc {val_acc:.4f} best {best_val:.4f}')
    model.load_state_dict(torch.load(ckpt, map_location=b.x.device))
    model.eval()
    return {'best_val_accuracy': best_val, 'best_epoch': best_epoch}


@torch.no_grad()
def evaluate_batched(model, b: Bundle):
    """All test documents inserted into the graph at once."""
    model.eval()
    logits = model(b.g_full)[b.test_ids].cpu().numpy()
    return classification_metrics(logits, b.labels[b.test_ids].cpu().numpy()), logits


@torch.no_grad()
def evaluate_strict(model, b: Bundle, doc_ids=None):
    """Strict per-document protocol (the paper's main protocol): each test document
    is inserted ALONE into the frozen train+valid graph, so no test document can
    affect another one's prediction."""
    model.eval()
    comp = b.comp
    doc_ids = comp['test_ids'] if doc_ids is None else doc_ids
    logits = np.zeros((len(doc_ids), 2))
    for k, i in enumerate(tqdm(doc_ids, desc='strict eval', leave=False)):
        g = to_pyg(single_doc_adj(comp, i, b.ww), b.x, b.x.device, b.edge_norm)
        logits[k] = model(g)[i].cpu().numpy()
    y = np.asarray(comp['labels'])[doc_ids]
    return classification_metrics(logits, y), logits


def evaluate(model, b: Bundle, strict=True):
    res, batched_logits = evaluate_batched(model, b)
    out = {f'{k}_batched': v for k, v in res.items()}
    if strict and not b.comp['config'].get('transductive'):
        res_s, strict_logits = evaluate_strict(model, b)
        out.update({f'{k}_strict': v for k, v in res_s.items()})
        out['batched_vs_strict_flips'] = int((batched_logits.argmax(1) != strict_logits.argmax(1)).sum())
    return out
