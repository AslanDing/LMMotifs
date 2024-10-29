import torch
from torch_geometric.data import Data
import numpy as np
import pickle

dataset_name = 'pubmed_gpt-4-1_5parser'
path = f'./data/ours/trees/{dataset_name}.pickle'

with open(path,'rb') as fp:
    data_pickle = pickle.load(fp)
    # print("xx")

datas = []
ori_tree = data_pickle['original_tree']
ori_feat = data_pickle['original_feature']
print("human: ")
for i in range(len(ori_tree)):
    tree = ori_tree[i]
    features = np.array(ori_feat[i])
    feature_0 = np.zeros_like(features[0,:])
    features = np.concatenate([feature_0.reshape(1,-1),features],axis=0)
    # print("xxx")
    edge_indexes = [[],[]]
    for t_dict in tree:
        parent = t_dict['parent']
        id = t_dict['id']
        edge_indexes[0].append(parent)
        edge_indexes[1].append(id)
        # edge_indexes[1].append(parent)
        # edge_indexes[0].append(id)
    edge_indexes = np.array(edge_indexes)
    if features.shape[0]!=edge_indexes.max()+1:
        print(features.shape[0],edge_indexes.max())
    data = Data(x=torch.from_numpy(features).to(torch.float),
                edge_index=torch.from_numpy(edge_indexes).to(torch.long),
                # edge_index=torch.tensor([[],[]]).to(torch.long),
                y=torch.tensor(0))
    datas.append(data)

sampled_tree = data_pickle['sampled_tree']
sampled_feat = data_pickle['sampled_feature']
print("AI: ")
for i in range(len(sampled_tree)):
    tree = sampled_tree[i]
    features = np.array(sampled_feat[i])
    feature_0 = np.zeros_like(features[0,:])
    features = np.concatenate([feature_0.reshape(1,-1),features],axis=0)
    # print("xxx")
    edge_indexes = [[],[]]
    for t_dict in tree:
        parent = t_dict['parent']
        id = t_dict['id']
        edge_indexes[0].append(parent)
        edge_indexes[1].append(id)
        # edge_indexes[1].append(parent)
        # edge_indexes[0].append(id)
    edge_indexes = np.array(edge_indexes)
    if features.shape[0] != edge_indexes.max() + 1:
        print(features.shape[0], edge_indexes.max())
    data = Data(x=torch.from_numpy(features).to(torch.float),
                edge_index=torch.from_numpy(edge_indexes).to(torch.long),
                # edge_index=torch.tensor([[],[]]).to(torch.long),
                y=torch.tensor(1))
    datas.append(data)

torch.save(datas,path.replace('trees','graphs').replace('.pickle','.pth'))
