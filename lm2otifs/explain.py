"""Motif extraction and explanation evaluation (Section 4.3, Section 5.3, Appendix E.2-E.3).

All explanations are computed on the strict per-document graph of a test document
(frozen train+valid graph plus that single document). Perturbations are applied as
an edge mask passed to the GCN as edge weights: weight 0 is equivalent to deleting
the edge, weight 1 leaves it unchanged (GCN degree normalisation uses the weights).

* ``gnnexplainer_edge_mask``   - GNNExplainer objective (prediction preservation +
                                 lambda * mask size + entropy) over an arbitrary edge set;
                                 used for local motifs over the document's one-hop
                                 neighbourhood (word-word and doc-word edges).
* ``corpus_motif_mask``        - one global mask over all word-word edges preserving the
                                 predictions on a set of training documents (corpus-level motifs).
* token attributors            - random / gradient x edge / LIME / KernelSHAP / GNNExplainer,
                                 each returning a ranking over the document's in-vocabulary
                                 tokens (unified protocol, Fig. 9-10, Table 22).
"""
import numpy as np
import torch
from tqdm import tqdm

from .graph import assemble_adj, single_doc_adj, to_pyg

BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]


# ---------------------------------------------------------------- graphs

class DocGraph:
    """Strict graph of one test document and the edge positions that belong to it."""

    def __init__(self, comp, x, doc_id, device, edge_norm='raw'):
        num_docs = comp['num_docs']
        adj = single_doc_adj(comp, doc_id)
        a = adj.tocoo()
        self.g = to_pyg(adj, x, device, edge_norm)
        self.doc_id = doc_id
        self.num_edges = a.row.shape[0]
        row, col = a.row.astype(np.int64), a.col.astype(np.int64)

        _, cols, _ = comp['test_dw_by_doc'][doc_id]
        token_wids = np.array(sorted({int(c) - num_docs for c in cols}), dtype=np.int64)
        in_doc = np.zeros(comp['number_nodes'] - num_docs, dtype=bool)
        in_doc[token_wids] = True
        r_w, c_w = row >= num_docs, col >= num_docs
        rw, cw = np.where(r_w, row - num_docs, 0), np.where(c_w, col - num_docs, 0)

        # word-word edges whose two endpoints are both tokens of this document
        self.ww_pos = np.where(r_w & c_w & in_doc[rw] & in_doc[cw])[0]
        # this document's doc-word edges (both orientations of the symmetric adjacency)
        self.dw_pos = np.where(((row == doc_id) & c_w & in_doc[cw]) |
                               ((col == doc_id) & r_w & in_doc[rw]))[0]
        # doc-word edge positions per token; tokens with idf 0 (present in every training
        # document) have no edge after symmetrisation and are dropped
        pos = {int(w): [] for w in token_wids}
        for e in self.dw_pos:
            pos[int(cw[e]) if row[e] == doc_id else int(rw[e])].append(int(e))
        self.tokens = [w for w in pos if pos[w]]
        self.token_pos = {w: pos[w] for w in self.tokens}

    def positions(self, tokens):
        out = []
        for w in tokens:
            out.extend(self.token_pos[w])
        return out


def masked_logits(model, g, mask):
    """Forward pass with an edge mask in [0, 1] (multiplied into the weights of a weighted model)."""
    w = mask if not model.weighted else g.edge_attr * mask
    return model(g, edge_weight=w)


@torch.no_grad()
def prob_with_mask(model, dg: DocGraph, drop_positions):
    mask = torch.ones(dg.num_edges, device=dg.g.edge_index.device)
    if len(drop_positions):
        mask[torch.as_tensor(np.asarray(drop_positions), dtype=torch.long, device=mask.device)] = 0.0
    out = masked_logits(model, dg.g, mask)
    return torch.softmax(out[dg.doc_id], dim=-1).cpu().numpy()


# ---------------------------------------------------------------- GNNExplainer

def _mask_loss(logp, m, edge_size, edge_ent):
    ent = -m * torch.log(m + 1e-15) - (1 - m) * torch.log(1 - m + 1e-15)
    return -logp + edge_size * m.mean() + edge_ent * ent.mean()


def gnnexplainer_edge_mask(model, dg: DocGraph, positions, epochs=150, lr=0.05,
                           edge_size=0.05, edge_ent=0.5, seed=0):
    """GNNExplainer with the learnable mask restricted to ``positions`` (other edges fixed
    at 1). ``edge_size`` is the size coefficient lambda of Sec. 4.3 / Fig. 7."""
    torch.manual_seed(seed)
    device = dg.g.edge_index.device
    idx = torch.as_tensor(np.asarray(positions), dtype=torch.long, device=device)
    ones = torch.ones(dg.num_edges, device=device)
    with torch.no_grad():
        pred = int(masked_logits(model, dg.g, ones)[dg.doc_id].argmax())
    logit_mask = torch.nn.Parameter(torch.randn(len(idx), device=device) * 0.1)
    opt = torch.optim.Adam([logit_mask], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        m = logit_mask.sigmoid()
        out = masked_logits(model, dg.g, ones.scatter(0, idx, m))
        logp = torch.log_softmax(out[dg.doc_id], dim=-1)[pred]
        _mask_loss(logp, m, edge_size, edge_ent).backward()
        opt.step()
    return logit_mask.detach().sigmoid().cpu().numpy()


def corpus_motif_mask(model, comp, x, device, doc_ids, epochs=300, lr=0.05,
                      edge_size=1e-4, edge_ent=0.1, seed=0, edge_norm='raw'):
    """Corpus-level motif: ONE mask over all word-word edges of the training graph,
    optimised to preserve the model's predictions on ``doc_ids`` (training documents,
    optionally of one class). Returns one score per entry of ``comp['ww']``."""
    torch.manual_seed(seed)
    num_docs = comp['num_docs']
    adj = assemble_adj(comp['number_nodes'], comp['ww'], comp['train_dw'])
    a = adj.tocoo()
    g = to_pyg(adj, x, device, edge_norm)
    E = a.row.shape[0]
    r, c = a.row, a.col
    ww_pos = np.where((r >= num_docs) & (c >= num_docs))[0]
    idx = torch.as_tensor(ww_pos, dtype=torch.long, device=device)
    docs = torch.as_tensor(doc_ids, dtype=torch.long, device=device)
    ones = torch.ones(E, device=device)
    model.eval()
    with torch.no_grad():
        pred = masked_logits(model, g, ones)[docs].argmax(-1)
    logit_mask = torch.nn.Parameter(torch.randn(len(ww_pos), device=device) * 0.1)
    opt = torch.optim.Adam([logit_mask], lr=lr)
    for _ in tqdm(range(epochs), desc='corpus motif', leave=False):
        opt.zero_grad()
        m = logit_mask.sigmoid()
        out = masked_logits(model, g, ones.scatter(0, idx, m))
        logp = torch.log_softmax(out[docs], dim=-1).gather(1, pred[:, None]).mean()
        _mask_loss(logp, m, edge_size, edge_ent).backward()
        opt.step()
    sym = logit_mask.detach().sigmoid().cpu().numpy()

    # map the symmetric-edge scores back onto the (row, col, pmi) triples of comp['ww']
    best = {}
    for e, v in zip(ww_pos, sym):
        key = (int(min(r[e], c[e])), int(max(r[e], c[e])))
        best[key] = max(best.get(key, 0.0), float(v))
    rows, cols, _ = comp['ww']
    return np.array([best.get((min(i, j), max(i, j)), 0.0) for i, j in zip(rows, cols)])


# ---------------------------------------------------------------- token attributors

def attr_random(dg, rng, **kw):
    return rng.rand(len(dg.tokens))


def attr_gradient(model, dg, **kw):
    """Gradient x edge: |d logit / d edge weight| on the document's doc-word edges, summed per token."""
    mask = torch.ones(dg.num_edges, device=dg.g.edge_index.device, requires_grad=True)
    out = masked_logits(model, dg.g, mask)
    out[dg.doc_id, int(out[dg.doc_id].argmax())].backward()
    grad = mask.grad.detach().abs().cpu().numpy()
    return np.array([grad[dg.token_pos[w]].sum() for w in dg.tokens])


def attr_gnnexplainer(model, dg, seed=0, **kw):
    """GNNExplainer mask scoped to the document's doc-word edges, max-pooled per token."""
    pos = dg.positions(dg.tokens)
    imp = dict(zip(pos, gnnexplainer_edge_mask(model, dg, pos, seed=seed)))
    return np.array([max(imp[p] for p in dg.token_pos[w]) for w in dg.tokens])


def _sampling_attr(model, dg, kernel, rng, n_samples=400):
    U = len(dg.tokens)
    pred = int(np.argmax(prob_with_mask(model, dg, [])))
    Z = np.ones((n_samples, U))
    y, w = np.zeros(n_samples), np.zeros(n_samples)
    sizes = np.arange(1, U)
    shap_kernel = (U - 1) / (sizes * (U - sizes))
    for s in range(n_samples):
        if kernel == 'lime':
            z = rng.rand(U) > 0.5
        else:  # KernelSHAP: coalition size drawn from the Shapley kernel
            z = np.zeros(U, dtype=bool)
            z[rng.choice(U, size=rng.choice(sizes, p=shap_kernel / shap_kernel.sum()),
                         replace=False)] = True
        Z[s] = z
        y[s] = prob_with_mask(model, dg, dg.positions([dg.tokens[j] for j in range(U) if not z[j]]))[pred]
        w[s] = np.exp(-((1.0 - z.sum() / U) ** 2) / 0.25) if kernel == 'lime' else 1.0
    X = np.hstack([Z, np.ones((n_samples, 1))])
    W = np.diag(w)
    coef = np.linalg.solve(X.T @ W @ X + 1e-6 * np.eye(U + 1), X.T @ W @ y)
    return coef[:U]


def attr_lime(model, dg, rng, **kw):
    return _sampling_attr(model, dg, 'lime', rng)


def attr_shap(model, dg, rng, **kw):
    return _sampling_attr(model, dg, 'shap', rng)


TOKEN_ATTRIBUTORS = {'random': attr_random, 'gradient': attr_gradient, 'lime': attr_lime,
                     'shap': attr_shap, 'gnnexplainer': attr_gnnexplainer}


# ---------------------------------------------------------------- MoRF / LeRF

def deletion_curves(model, dg, items, scores, pos_of, budgets=BUDGETS):
    """MoRF / LeRF / sufficiency curves for a ranking over ``items``.

    ``pos_of(selected_items)`` returns the edge positions to delete. MoRF deletes the
    top-k items, LeRF the bottom-k items, sufficiency keeps only the top-k items.
    Values are the probability of the originally predicted class, plus whether the
    prediction is still correct.
    """
    p_full = prob_with_mask(model, dg, [])
    pred = int(np.argmax(p_full))
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    n = len(items)
    res = {'p_full': float(p_full[pred]), 'pred': pred, 'n_items': n,
           'morf': [], 'lerf': [], 'suff': [], 'morf_pred': [], 'lerf_pred': []}
    for b in budgets:
        k = max(1, int(round(b * n)))
        top = [items[j] for j in order[:k]]
        bottom = [items[j] for j in order[-k:]]
        rest = [items[j] for j in order[k:]]
        p_m = prob_with_mask(model, dg, pos_of(top))
        p_l = prob_with_mask(model, dg, pos_of(bottom))
        res['morf'].append(float(p_m[pred]))
        res['lerf'].append(float(p_l[pred]))
        res['suff'].append(float(prob_with_mask(model, dg, pos_of(rest))[pred]))
        res['morf_pred'].append(int(np.argmax(p_m)))
        res['lerf_pred'].append(int(np.argmax(p_l)))
    return res


def summarize_curves(per_doc, labels=None):
    """Average curves; AOPC = mean drop over the budget grid."""
    p_full = np.array([r['p_full'] for r in per_doc])[:, None]
    morf = np.array([r['morf'] for r in per_doc])
    lerf = np.array([r['lerf'] for r in per_doc])
    suff = np.array([r['suff'] for r in per_doc])
    out = {'n_docs': len(per_doc),
           'p_full_mean': float(p_full.mean()),
           'morf_curve': morf.mean(0).tolist(),
           'lerf_curve': lerf.mean(0).tolist(),
           'suff_curve': suff.mean(0).tolist(),
           'comprehensiveness_aopc': float((p_full - morf).mean()),
           'sufficiency_aopc': float((p_full - suff).mean()),
           'morf_lerf_gap_aopc': float((lerf - morf).mean())}
    if labels is not None:
        y = np.asarray(labels)[:, None]
        out['morf_accuracy_curve'] = (np.array([r['morf_pred'] for r in per_doc]) == y).mean(0).tolist()
        out['lerf_accuracy_curve'] = (np.array([r['lerf_pred'] for r in per_doc]) == y).mean(0).tolist()
    return out
