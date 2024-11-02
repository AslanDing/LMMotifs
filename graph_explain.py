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


dataset_name = 'pubmed_gpt-4-1'
path = f'./data/ours/trees/{dataset_name}.pickle'
with open(path,'rb') as fp:
    data_pickle = pickle.load(fp)
oritinal_tree = data_pickle['original_tree']
sampled_tree = data_pickle['sampled_tree']
all_trees = []
all_trees.extend(oritinal_tree)
all_trees.extend(sampled_tree)

path = f'./data/ours/graphs/{dataset_name}.pth'
graphs = torch.load(path)
# explain_graphs = []
# explain_graphs.extend()
loader = DataLoader(graphs, batch_size=1, shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch_geometric.nn.global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x
model = GCN(in_channels=300, hidden_channels=100, out_channels=2)  # 2 classes
path = f'./data/ours/model/{dataset_name}.pth'
model.load_state_dict(torch.load(path,map_location='cpu'))

save_dir = f'./data/ours/explanations/{dataset_name}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def visualization(graph,text_id, edge_weights,index):
    nodes = {}
    G = nx.DiGraph()
    nodes[0] = 'root'
    for tmp_dict in text_id:
        nodes[tmp_dict['id']] = tmp_dict['text']

    for node, label in nodes.items():
        G.add_node(node, label=label)

    edge_lists = graph.edge_index.cpu().numpy()

    for (i,j),w in zip(edge_lists.T,edge_weights):
        G.add_edge(i, j, weight=w)

    # pos = nx.spring_layout(G)
    pos = nx.nx_agraph.graphviz_layout(G, prog='twopi')

    edge_weights = nx.get_edge_attributes(G, 'weight')
    # Normalize the edge weights to map them to a colormap (e.g., 'viridis')
    weight_values = list(edge_weights.values())
    norm = plt.Normalize(min(weight_values), max(weight_values))
    cmap = cm.viridis  # You can change this to another colormap if needed
    edge_colors = [cmap(norm(weight)) for weight in weight_values]

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels=nodes, font_size=7)
    # nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray')
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color=edge_colors,arrowstyle='->',arrows=True, connectionstyle="arc3,rad=0.1")

    edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.title("Graph Visualization with Node Labels and Edge Weights")
    plt.axis('off')  # Turn off the axis
    # plt.show()
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label="Edge Weight")  # Show colorbar
    plt.savefig(save_dir+'/'+f'{dataset_name}_{index}.png')
    # plt.savefig(path.replace('model','explanations').replace('.pth',f'_{index}.png'))


def explain():
    device ='cuda:7'
    explainer = GNNExplainer(device,model)
    for idx,data in enumerate(loader):
        if idx >=10 and idx <150:
            continue
        if idx >=160:
            continue
        data.to(device)
        y = data.y
        out = model(data)
        edge_imp = explainer.explain_graph(data)
        # acc_list = explainer.evaluate_acc([0.1 * i for i in range(1, 11)])[0]
        # print("xx")
        visualization(data,all_trees[idx],edge_imp,idx)


if __name__=="__main__":
    explain()


