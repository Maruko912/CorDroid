import dgl as dgl
import torch.utils.data as data
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import scipy.sparse as sp

def image_loader(path):
    normalize = transforms.Normalize(mean=[0.20],std=[0.19])
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    img_pil =  Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor

def default_loader(feature):
    return torch.from_numpy(feature)

def construct_graph(adj_mat, node_feature):
    g = dgl.from_scipy(adj_mat)
    g = dgl.add_self_loop(g)
    g.ndata['h'] = torch.from_numpy(node_feature)
    return g

def collate(samples):

    graphs = [item[0] for item in samples]
    imgs = [item[1] for item in samples]
    labels = [item[2] for item in samples]
    imgs = torch.stack(imgs)

    return dgl.batch(graphs), imgs, torch.tensor(labels, dtype=torch.long)

class Mydataset(data.Dataset):
    def __init__(self, all_data, loader=default_loader, construct_graph=construct_graph, image_loader=image_loader):
        self.all_data = all_data
        self.loader = loader
        self.image_loader = image_loader
        self.construct_graph = construct_graph

    def __getitem__(self, index):

        adj_path = os.path.join(self.all_data[index][0], 'sen_adj_calls.npz')
        node_features_path = os.path.join(self.all_data[index][0], 'sen_node_features.npy')
        img_path = os.path.join(self.all_data[index][0], "opcode_img.png")
        img_tensor = self.image_loader(img_path)
        adj_matrix = sp.load_npz(adj_path)
        node_features = np.load(node_features_path)

        graph = self.construct_graph(adj_matrix, node_features)
        if adj_matrix.shape[0]==0:
            adj_matrix = sp.coo_matrix(np.zeros((1,1)))
            node_features = np.ones((1,100),dtype=np.float32)
            graph = self.construct_graph(adj_matrix, node_features)
        target = self.all_data[index][1]
        return graph, img_tensor, target


    def __len__(self):
        return len(self.all_data)
