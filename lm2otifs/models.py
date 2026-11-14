"""Detectors over the corpus graph (Section 4.2).

``TextGCN``  - the default detector: two GCN layers |V| -> 64 -> 2 applied to the
               whole corpus graph; document nodes are classified (Appendix D.2).
``SweepGNN`` - the uniform family used for the architecture sweep of Table 2:
               a sparse input projection followed by two message-passing layers
               of one type (gcn / gat / sage / transformer / pna).

Both return logits for every node. ``weighted=False`` (default, "uw") consumes the
binarised adjacency; ``weighted=True`` ("w") consumes the edge values.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, PNAConv, SAGEConv, TransformerConv
from torch_geometric.utils import degree


def make_act(name):
    return {'silu': nn.SiLU(), 'relu': nn.ReLU(), 'identity': nn.Identity()}[name]


class TextGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_classes=2, weighted=False,
                 act='silu', dropout=True):
        super().__init__()
        self.weighted = weighted
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.act = make_act(act)
        # as in the original pipeline, dropout is applied to the input of the second layer only
        self.dropout = nn.Dropout() if dropout else nn.Identity()

    def forward(self, g, edge_weight=None):
        """``edge_weight`` overrides the graph's own weights (used by the explainers,
        which pass a [0, 1] edge mask; a zero weight is equivalent to deleting the edge)."""
        if edge_weight is None and self.weighted:
            edge_weight = g.edge_attr
        h = self.act(self.conv1(g.x, g.edge_index, edge_weight))
        h = self.dropout(h)
        return self.conv2(h, g.edge_index, edge_weight)


class SparseInputProj(nn.Module):
    """One-hot -> hidden projection; equivalent to the first GCN weight matrix on
    one-hot input, and needed by architectures that aggregate before projecting."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        if x.is_sparse:
            return torch.sparse.mm(x, self.weight) + self.bias
        return x @ self.weight + self.bias


ARCHS = ['gcn', 'gat', 'sage', 'transformer', 'pna']
WEIGHTED_SUPPORT = {'gcn', 'gat', 'transformer', 'pna'}


def pna_degree_histogram(edge_index, num_nodes):
    d = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)
    return torch.bincount(d)


def make_conv(arch, in_dim, out_dim, weighted, deg=None):
    edge_kw = {'edge_dim': 1} if weighted else {}
    if arch == 'gcn':
        return GCNConv(in_dim, out_dim)
    if arch == 'sage':
        return SAGEConv(in_dim, out_dim)
    if arch == 'gat':
        return GATConv(in_dim, out_dim, **edge_kw)
    if arch == 'transformer':
        return TransformerConv(in_dim, out_dim, **edge_kw)
    if arch == 'pna':
        return PNAConv(in_dim, out_dim, aggregators=['mean', 'min', 'max', 'std'],
                       scalers=['identity', 'amplification', 'attenuation'], deg=deg, **edge_kw)
    raise ValueError(f'unknown architecture {arch}')


class SweepGNN(nn.Module):
    def __init__(self, arch, in_dim, hidden_dim=64, num_classes=2, weighted=False,
                 deg=None, act='silu', dropout=True):
        super().__init__()
        if weighted and arch not in WEIGHTED_SUPPORT:
            raise ValueError(f'{arch} does not support weighted edges')
        self.arch, self.weighted = arch, weighted
        self.proj = SparseInputProj(in_dim, hidden_dim)
        self.conv1 = make_conv(arch, hidden_dim, hidden_dim, weighted, deg)
        self.conv2 = make_conv(arch, hidden_dim, num_classes, weighted, deg)
        self.act = make_act(act)
        self.dropout = nn.Dropout() if dropout else nn.Identity()

    def _conv(self, conv, h, g):
        if not self.weighted:
            return conv(h, g.edge_index)
        if self.arch == 'gcn':
            return conv(h, g.edge_index, edge_weight=g.edge_attr)
        return conv(h, g.edge_index, edge_attr=g.edge_attr.view(-1, 1))

    def forward(self, g, edge_weight=None):
        if edge_weight is not None:
            raise NotImplementedError('edge masks are only supported by TextGCN')
        h = self.proj(g.x)
        h = self.dropout(h)
        h = self.act(self._conv(self.conv1, h, g))
        return self._conv(self.conv2, h, g)
