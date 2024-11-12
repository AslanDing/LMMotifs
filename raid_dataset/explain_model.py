import os.path

import numpy as np
import torch
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import pickle
from explainers.gnn_explainer import GNNExplainer
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import torch
from torch.optim.swa_utils import AveragedModel
from graph_build import build_text_graph_dataset1
from model import TextGNN
from eval import eval,MovingAverage

from pprint import pprint

from torch_geometric.utils.subgraph import subgraph,k_hop_subgraph
from transformers import BertTokenizer
from collections import OrderedDict

model_name = 'gchhablani/bert-base-cased-finetuned-sst2'  # Choose your preferred model name
# model_name = 'gchhablani/bert-base-cased-finetuned-sst2' # 0.92
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
vocab_dict = tokenizer.vocab

reversed_dict = OrderedDict()
for key, val in vocab_dict.items():
    reversed_dict[val] = key
print("")

def visualization(graph, dataset, node_lists, edge_weights,index,
                  save_dir = '../data/raid/logs/explanations'):
    labels = dataset.labels
    node_to_ids = dataset.vocab

    start_node = graph.x.shape[0] - len(node_to_ids)

    nodes = {}
    G = nx.Graph()
    nodes[index] = f'doc_{labels[index]}'
    # for idx in range(start_node):
    #     nodes[idx] = f'doc_{labels[idx]}'

    for idx in range(start_node,graph.x.shape[0]):
        # nodes[tmp_dict['id']] = tmp_dict['text']
        nodes[idx] = reversed_dict[node_to_ids[idx - start_node]]

    for node, label in nodes.items():
        G.add_node(node, label=label)

    edge_lists = graph.edge_index.cpu().numpy()

    # max_value = edge_weights.max()
    # min_value = edge_weights.min()
    percentile_99 = np.percentile(edge_weights, 99.5)
    for (i,j),w in zip(edge_lists.T,edge_weights):
        if w<percentile_99:
            continue
        else:
            G.add_edge(i, j, weight=w)

    connected_components = list(nx.connected_components(G))
    # Access the first connected component (as a set of nodes)
    first_component = connected_components[0]

    # Create a subgraph from the connected component
    G = G.subgraph(first_component)

    pos = nx.spring_layout(G)
    # pos = nx.nx_agraph.graphviz_layout(G, prog='twopi')
    edge_weights = nx.get_edge_attributes(G, 'weight')
    # Normalize the edge weights to map them to a colormap (e.g., 'viridis')
    weight_values = list(edge_weights.values())
    norm = plt.Normalize(min(weight_values), max(weight_values))
    cmap = cm.viridis  # You can change this to another colormap if needed
    edge_colors = [cmap(norm(weight)) for weight in weight_values]

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue') #nodelist=node_lists,
    new_nodes = {n:nodes[n] for n in G}
    nx.draw_networkx_labels(G, pos, labels=new_nodes, font_size=7)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color=edge_colors)
    # nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color=edge_colors,arrowstyle='->',arrows=True, connectionstyle="arc3,rad=0.1")

    edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.title("Graph Visualization with Node Labels and Edge Weights")
    plt.axis('off')  # Turn off the axis
    # plt.show()
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label="Edge Weight")  # Show colorbar
    plt.savefig(save_dir+'/'+f'raid_{index}.png')
    # plt.savefig(path.replace('model','explanations').replace('.pth',f'_{index}.png'))


def explain():
    device = 'cuda:1'

    dataset_dir = '../data/raid/subset'
    textdataset =  build_text_graph_dataset1(dataset_dir,window_size=20)
    train_dataset, val_dataset, test_dataset = textdataset.tvt_split()

    train_dataset.init_node_feats('one_hot_init', device)
    val_dataset.init_node_feats('one_hot_init', device)

    # build model
    model = TextGNN(pred_type='softmax', node_embd_type='gcn',
            num_layers=2, layer_dim_list=[55698,64,2], act='silu',
            bn=True, num_labels=2, class_weights=True, dropout=True)
    model.to(device)
    model_ema = AveragedModel(model)
    model_ema.load_state_dict(torch.load('../data/raid/logs/model.pth',map_location=device))
    model = model_ema.module
    pyg_graph = train_dataset.get_pyg_graph(device)

    save_dir = f'../data/raid/logs/explanations'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    explainer = GNNExplainer(device,model)

    count = 0
    for i in range(len(train_dataset.node_ids)):

        if count>=10:
            break
        if train_dataset.labels[i]!=0:
            continue

        idx = train_dataset.node_ids[i]

        subset, edge_index, inv, edge_mask = k_hop_subgraph(node_idx=idx,num_hops=1,edge_index=pyg_graph.edge_index)

        all_nodes = np.arange(pyg_graph.x.shape[0]).tolist()
        word_nodes = set(all_nodes) - set(train_dataset.node_ids) \
                     - set(test_dataset.node_ids) - set(val_dataset.node_ids)
        subset_nodes = [idx]
        subset_nodes.extend(list(word_nodes))

        new_set = set(subset.cpu().numpy().tolist()).intersection(set(subset_nodes))
        new_set = list(new_set)
        edge_index, edge_attr = subgraph(new_set,pyg_graph.edge_index,pyg_graph.edge_attr)
        new_pyg_graph = Data(x=pyg_graph.x,edge_index=edge_index,edge_attr=edge_attr)
        edge_imp = explainer.explain_nodes(new_pyg_graph,idx=idx)

        visualization(new_pyg_graph,textdataset, new_set, edge_imp, idx)
        count+=1

    count = 0
    for i in range(len(train_dataset.node_ids)):

        if count>=10:
            break

        if train_dataset.labels[i] != 1:
            continue

        idx = train_dataset.node_ids[i]

        subset, edge_index, inv, edge_mask = k_hop_subgraph(node_idx=idx, num_hops=1, edge_index=pyg_graph.edge_index)

        all_nodes = np.arange(pyg_graph.x.shape[0]).tolist()
        word_nodes = set(all_nodes) - set(train_dataset.node_ids) \
                     - set(test_dataset.node_ids) - set(val_dataset.node_ids)
        subset_nodes = [idx]
        subset_nodes.extend(list(word_nodes))

        new_set = set(subset.cpu().numpy().tolist()).intersection(set(subset_nodes))
        new_set = list(new_set)
        edge_index, edge_attr = subgraph(new_set, pyg_graph.edge_index, pyg_graph.edge_attr)
        new_pyg_graph = Data(x=pyg_graph.x, edge_index=edge_index, edge_attr=edge_attr)
        edge_imp = explainer.explain_nodes(new_pyg_graph, idx=idx)

        visualization(new_pyg_graph, textdataset, new_set, edge_imp, idx)
        count += 1

if __name__=="__main__":
    explain()