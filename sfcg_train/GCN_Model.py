import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn as nn

class GCN_Model(nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, n_classes):
        super(GCN_Model,self).__init__()
        self.conv1 = GraphConv(input_dim,hidden_dim1, activation=nn.ReLU(inplace=True))
        self.conv2 = GraphConv(hidden_dim1,hidden_dim2, activation=nn.ReLU(inplace=True))
        self.dense1 = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim3,hidden_dim4),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(hidden_dim4, n_classes)
        )

    def extract_feature(self, x):
        h = x.ndata['h'].float()
        h1 = self.conv1(x, h)
        h2 = self.conv2(x, h1)
        x.ndata['h'] = h2

        hg = dgl.mean_nodes(x, 'h')
        return hg

    def forward(self, x):
        hg = self.extract_feature(x)
        hg2 = self.dense1(hg)
        out = self.dense2(hg2)
        return out