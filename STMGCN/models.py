import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.cluster import KMeans
import torch.optim as optim
from random import shuffle
import pandas as pd
import numpy as np
import scanpy as sc
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self,nfeat,out):
        super(GCN, self).__init__()
        self.gc = GraphConvolution(nfeat,out)
        

    def forward(self,x,adj):
        x =self.gc(x,adj)
        return x

class Attention(nn.Module):
    def __init__(self,in_size,hidden_size=16):
        super(Attention, self).__init__()
        self.project=nn.Linear(in_size, 1, bias=False)

    def forward(self,z):
        w = self.project(z)
        beta = torch.softmax(w,dim=1)
        return (beta*z).sum(1),beta  


class MGCN(nn.Module):
    def __init__(self,nfeat,nclass,nhid1):
        super(MGCN, self).__init__()
        self.GCNA1 = GCN(nfeat,nhid1)
        self.GCNA2 = GCN(nfeat,nhid1)
        self.attention = Attention(nhid1)


    def forward(self,x,adj1,adj2):
   
        emb1 = self.GCNA1(x,adj1)
        emb2 = self.GCNA2(x,adj2)
        emb = torch.stack([emb1,emb2],dim=1)
        emb,_ = self.attention(emb)  
        return emb
   

class STMGCN(nn.Module):
    def __init__(self,nfeat,nclass,nhid1):
        super(STMGCN, self).__init__()
        self.mgcn = MGCN(nfeat,nclass,nhid1)
        self.cluster_layer = Parameter(torch.Tensor(nclass,nhid1))
        torch.nn.init.xavier_normal_(self.cluster_layer)




    def forward(self,x,adj1,adj2): 
        x = self.mgcn(x,adj1,adj2)
        self.alpha = 0.2
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.cluster_layer)**2, dim=2) / self.alpha))
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x,q




