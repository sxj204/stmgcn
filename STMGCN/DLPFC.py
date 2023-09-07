# -*- coding:utf-8 -*-

import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from loss import target_distribution, kl_loss
import torch.optim as optim
from torch.nn.parameter import Parameter
from anndata import AnnData
import torch
from sklearn.cluster import KMeans
from util import *
import torch.nn as nn
import argparse
from sklearn.decomposition import PCA
from models import *



def train_MSpaGCN(opts):
    if opts.dataset == 'DLPFC' :
        features_adata,features,labels = load_data(opts.dataset,opts.sicle,opts.npca)
    else:
        features_adata,features,labels = load_Medata(opts.dataset,opts.npca)
    adj1,adj2 = load_graph(opts.dataset,opts.sicle,opts.l)

    model =STMGCN(nfeat=features.shape[1],
                    nhid1=opts.nhid1,
                    nclass=opts.n_cluster
                    )
    if opts.cuda:
        model.cuda()
        features = features.cuda()
        adj1 = adj1.cuda()
        adj2 = adj2.cuda()

    optimizer = optim.Adam(model.parameters(),lr=opts.lr, weight_decay=opts.weight_decay)
    emb = model.mgcn(features,adj1,adj2)


    if opts.initcluster == "kmeans":
        print("Initializing cluster centers with kmeans, n_clusters known")
        n_clusters=opts.n_cluster
        kmeans = KMeans(n_clusters,n_init=20)
        y_pred = kmeans.fit_predict(emb.detach().cpu().numpy())

    elif opts.initcluster == "louvain":
        print("Initializing cluster centers with louvain,resolution=",opts.res)
        adata=sc.AnnData(emb.detach().cpu().numpy())
        sc.pp.neighbors(adata, n_neighbors=opts.n_neighbors)
        sc.tl.louvain(adata,resolution=opts.res)
        y_pred=adata.obs['louvain'].astype(int).to_numpy()
        n=len(np.unique(y_pred))



    emb=pd.DataFrame(emb.detach().cpu().numpy(),index=np.arange(0,emb.shape[0]))
    Group=pd.Series(y_pred,index=np.arange(0,emb.shape[0]),name="Group")
    Mergefeature=pd.concat([emb,Group],axis=1)
    cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())

    y_pred_last = y_pred
    with torch.no_grad():
        model.cluster_layer.copy_(torch.tensor(cluster_centers))

    #模型训练
    model.train()
    for epoch in range(opts.max_epochs):

        if epoch % opts.update_interval == 0:
            _, tem_q = model(features,adj1,adj2)
            tem_q = tem_q.detach()
            p = target_distribution(tem_q)

            y_pred = torch.argmax(tem_q, dim=1).cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            y = labels

            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
            print('Iter {}'.format(epoch), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            if epoch>0 and delta_label < opts.tol:
                print('delta_label ', delta_label, '< tol ', opts.tol)
                print("Reach tolerance threshold. Stopping training.")
                break

        optimizer.zero_grad()
        x,q = model(features,adj1,adj2)
        loss = kl_loss(q.log(), p)
        loss.backward()
        optimizer.step()


    #save emnddings
    key_added = "STMGCN"
    embeddings = pd.DataFrame(x.detach().cpu().numpy())
    embeddings.index = features_adata.obs_names
    features_adata.obsm[key_added] = embeddings.loc[features_adata.obs_names,].values
  
    #plot spatial
    plt.rcParams["figure.figsize"] = (6, 3)
    sc.pl.spatial(features_adata,color=["pred", "Ground Truth"], title=['STMGCN (ARI=%.3f)' % ARI,'Ground Truth'],na_in_legend = False,show=True)












def parser_set():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--nhid1',type=int,default=32)
    parser.add_argument('--n_cluster',default=7,type=int)
    parser.add_argument('--max_epochs',default=2000,type=int)
    parser.add_argument('--update_interval',default= 3,type=int)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--weight_decay',default=0.001,type=float)
    parser.add_argument('--dataset', type=str, default='DLPFC')
    parser.add_argument('--sicle', default="151673", type=str)
    parser.add_argument('--tol', default=0.0001, type=float)
    parser.add_argument('--l', default=1, type=float)
    parser.add_argument('--npca', default=50, type=int)
    parser.add_argument('--n_neighbors',type=int,default=10)
    parser.add_argument('--initcluster', default="kmeans", type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    return args

if __name__ == "__main__":
    opts = parser_set()
    print(opts)
    train_MSpaGCN(opts)

























































