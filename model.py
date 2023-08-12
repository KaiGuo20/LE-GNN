import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import wandb
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot
from sklearn.metrics.pairwise import cosine_similarity
from layers import *
device = torch.device("cuda:0")
import torch

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())




class GraphConvolution_LEGNN(torch.nn.Module):
    def __init__(self, in_features, out_features, al, all, adj, residual = False):
        super(GraphConvolution_LEGNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.al = al
        self.all = all

    def forward(self, input, A, h0, y):
        adjacency_mask = torch.mm(y, y.t())
        adj = A.to_dense()
        adj = adj * adjacency_mask
        adj = F.normalize(adj, p=1, dim=1)
        adj = to_sparse(adj)
        support = torch.sparse.mm(adj, input)
        output = (1 - self.al) * support + self.al * h0
        if self.residual:
            output = output + 0*input
        return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'





if __name__ == '__main__':
    pass



class LEGNN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, al, all, adj, norm):
        super(LEGNN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution_LEGNN(nhidden, nhidden, al, all, adj))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nhidden = nhidden
        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)
        self.norm = norm
        self.adj = adj
    def forward(self, x, adj,y_label, idx_train):
        self.idx_train = idx_train
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training).cuda()
        inner = self.fcs[0](x)
        layer_inner = self.act_fn(inner)
        Pseudo = self.fcs[1](layer_inner)
        _layers.append(layer_inner)
        y_hat = Pseudo
        y_hat[idx_train] = y_hat[idx_train] + 0.1* y_label[idx_train]
        y_hat = F.softmax(y_hat,dim=1)
        for i,con in enumerate(self.convs):
            layer_inner = F .dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = con(layer_inner, adj, _layers[0], y_hat)
            if self.norm == 'True':
                layer_inner = F.normalize(layer_inner, p='fro', dim=1)#2
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = 1*self.fcs[-1](layer_inner)

        return F.log_softmax(layer_inner, dim=1), Pseudo




