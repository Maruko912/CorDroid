import dgl as dgl
import torch.utils.data as data
import numpy as np
import os
import torch
import scipy.sparse as sp

def default_loader(feature):
    return torch.from_numpy(feature)

def construct_graph(adj_mat, node_feature):
    g = dgl.from_scipy(adj_mat)
    g = dgl.add_self_loop(g)
    g.ndata['h'] = torch.from_numpy(node_feature)
    return g

def collate(samples):

    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

class Mydataset(data.Dataset):
    def __init__(self, all_data, loader=default_loader, construct_graph=construct_graph):
        self.all_data = all_data
        self.loader = loader
        self.construct_graph = construct_graph

    def __getitem__(self, index):

        adj_path = os.path.join(self.all_data[index][0], 'sen_adj_calls.npz')
        node_features_path = os.path.join(self.all_data[index][0], 'sen_node_features.npy')
        adj_matrix = sp.load_npz(adj_path)
        node_features = np.load(node_features_path)

        graph = self.construct_graph(adj_matrix, node_features)
        target = self.all_data[index][1]
        return graph, target
        
    def __len__(self):
        return len(self.all_data)
