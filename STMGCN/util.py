import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy
import os

import sklearn
from sklearn.decomposition import PCA
from anndata import AnnData,read_csv,read_text,read_mtx
from scanpy import read_10x_h5
from scipy.sparse import issparse
import random
import torch
from calculate_adj import calculate_adj_matrix




def calculate_p(adj, l):
    adj_exp=np.exp(-1*(adj**2)/(2*(l**2)))
    return np.mean(np.sum(adj_exp,1))-1



def find_meta_gene(input_adata,
                    pred,
                    target_domain,
                    start_gene,
                    mean_diff=0,
                    early_stop=True,
                    max_iter=5,
                    use_raw=False):
    meta_name=start_gene
    adata=input_adata.copy()
    adata.obs["meta"]=adata.X[:,adata.var.index==start_gene]
    adata.obs["pred"]=pred
    num_non_target=adata.shape[0]
    for i in range(max_iter):
        #Select cells
        tmp=adata[((adata.obs["meta"]>np.mean(adata.obs[adata.obs["pred"]==target_domain]["meta"]))|(adata.obs["pred"]==target_domain))]
        tmp.obs["target"]=((tmp.obs["pred"]==target_domain)*1).astype('category').copy()
        if (len(set(tmp.obs["target"]))<2) or (np.min(tmp.obs["target"].value_counts().values)<5):
            print("Meta gene is: ", meta_name)
            return meta_name, adata.obs["meta"].tolist()
        #DE
        sc.tl.rank_genes_groups(tmp, groupby="target",reference="rest", n_genes=1,method='wilcoxon')
        adj_g=tmp.uns['rank_genes_groups']["names"][0][0]
        add_g=tmp.uns['rank_genes_groups']["names"][0][1]
        meta_name_cur=meta_name+"+"+add_g+"-"+adj_g
        print("Add gene: ", add_g)
        print("Minus gene: ", adj_g)
        #Meta gene
        adata.obs[add_g]=adata.X[:,adata.var.index==add_g]
        adata.obs[adj_g]=adata.X[:,adata.var.index==adj_g]
        adata.obs["meta_cur"]=(adata.obs["meta"]+adata.obs[add_g]-adata.obs[adj_g])
        adata.obs["meta_cur"]=adata.obs["meta_cur"]-np.min(adata.obs["meta_cur"])
        mean_diff_cur=np.mean(adata.obs["meta_cur"][adata.obs["pred"]==target_domain])-np.mean(adata.obs["meta_cur"][adata.obs["pred"]!=target_domain])
        num_non_target_cur=np.sum(tmp.obs["target"]==0)
        if (early_stop==False) | ((num_non_target>=num_non_target_cur) & (mean_diff<=mean_diff_cur)):
            num_non_target=num_non_target_cur
            mean_diff=mean_diff_cur
            print("Absolute mean change:", mean_diff)
            print("Number of non-target spots reduced to:",num_non_target)
        else:
            print("Stopped!", "Previous Number of non-target spots",num_non_target, num_non_target_cur, mean_diff,mean_diff_cur)
            print("Previous Number of non-target spots",num_non_target, num_non_target_cur, mean_diff,mean_diff_cur)
            print("Previous Number of non-target spots",num_non_target)
            print("Current Number of non-target spots",num_non_target_cur)
            print("Absolute mean change", mean_diff)
            print("===========================================================================")
            print("Meta gene: ", meta_name)
            print("===========================================================================")
            return meta_name, adata.obs["meta"].tolist()
        meta_name=meta_name_cur
        adata.obs["meta"]=adata.obs["meta_cur"]
        print("===========================================================================")
        print("Meta gene is: ", meta_name)
        print("===========================================================================")
    return meta_name, adata.obs["meta"].tolist()


def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)
    


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  #其思想是 按照(row_index, column_index, value)的方式存储每一个非0元素，所以存储的数据结构就应该是一个以三元组为元素的列表List[Tuple[int, int, int]]
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)) #from_numpy()用来将数组array转换为张量Tensor vstack（）：按行在下边拼接
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):  
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0. 
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx) 
    return mx

def load_Medata(dataset,npca=20):

    if dataset == 'osmFISH':
        count_patch = os.path.join('../data', dataset, 'count.csv')
        count = pd.read_csv(count_patch, sep=',', header=None)
        adata = sc.AnnData(count)
        label = pd.read_csv('../data/' + dataset + '/labeltruth.txt', sep=',', header=None)
        label.columns = ['order','Ground Truth']  
        labels = pd.factorize(label["Ground Truth"].astype("category"))[0]
        l = label.iloc[:,1].tolist()
        adata.obs['Ground Truth'] = l
        pos = pd.read_csv('../data/' + dataset + '/pos.csv', sep=',', header=None)
        x_array = pos.iloc[:, 0]
        y_array = pos.iloc[:, 1]
        adata.obs['x_array'] = x_array
        adata.obs['y_array'] = y_array
    elif dataset == 'Mouse embryo data':
        count_patch = os.path.join('../data', dataset, 'count.csv')
        count = pd.read_csv(count_patch, sep=',', header=None)
        adata = sc.AnnData(count)
        label = pd.read_csv('../data/' + dataset + '/labeltruth.txt', sep=',', header=None)
        label.columns = ['Ground Truth'] 
        labels = pd.factorize(label["Ground Truth"].astype("category"))[0]
        adata.obs['Ground Truth'] = label.iloc[:,0].tolist()
        pos = pd.read_csv('../data/' + dataset + '/pos.csv', sep=',', header=None)
        x_array = pos.iloc[:, 0]
        y_array = pos.iloc[:, 1]
        adata.obs['x_array'] = x_array
        adata.obs['y_array'] = y_array
    elif dataset == 'Stereo-seq mouse olfactory bulb':
        counts_file = os.path.join('../data/Stereo-seq mouse olfactory bulb/RNA_counts.tsv')
        coor_file = os.path.join('../data/Stereo-seq mouse olfactory bulb/position.tsv')
        counts = pd.read_csv(counts_file, sep='\t', index_col=0)
        coor_df = pd.read_csv(coor_file, sep='\t')
        counts.columns = ['Spot_' + str(x) for x in counts.columns]
        coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.loc[:, ['x', 'y']]
        adata = sc.AnnData(counts.T)
        adata.var_names_make_unique()
        coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
        adata.obs['x'] = coor_df['x'].tolist()
        adata.obs['y'] = coor_df['y'].tolist()
        adata.obsm["spatial"] = coor_df.to_numpy()
        used_barcode = pd.read_csv(os.path.join('../data/Stereo-seq mouse olfactory bulb/used_barcodes.txt'), sep='\t', header=None)
        used_barcode = used_barcode[0]
        adata = adata[used_barcode,]
        # 过滤
        sc.pp.filter_genes(adata, min_cells=50)
        labels = 0

    prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    pca = PCA(n_components = npca)
    if issparse(adata.X):
        pca.fit(adata.X.A)  #fit(X)，表示用数据X来训练PCA模型。  函数返回值：调用fit方法的对象本身。比如pca.fit(X)，表示用X对pca这个对象进行训练。
        embed=pca.transform(adata.X.A)  #用X来训练PCA模型，同时返回降维后的数据,embed就是降维后的数据。
    else:
        pca.fit(adata.X)
        embed=pca.transform(adata.X)
    features = torch.FloatTensor(np.array(embed))
    return adata,features,labels



def load_data(dataset,silce,npca):
    if dataset=="DLPFC":
        input_dir = os.path.join('../data',dataset,silce)
        adata = sc.read_visium(path=input_dir, count_file=silce+'_filtered_feature_bc_matrix.h5')
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.columns = ['imagerow', 'imagecol']
        adata.obs['x_pixel'] = coor['imagecol'].tolist()
        adata.obs['y_pixel'] = coor['imagerow'].tolist()
        label = pd.read_csv(os.path.join('../data',dataset, silce, silce+'_truth.txt'), sep='\t', header=None, index_col=0)
        label.columns = ['Ground Truth']
        labels = pd.factorize(label["Ground Truth"].astype("category"))[0]
        adata.obs['Ground Truth'] = label
    else:
        input_dir = os.path.join('../data', dataset)
        adata = sc.read_visium(path=input_dir, count_file='V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5')
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.columns = ['imagerow', 'imagecol']
        adata.obs['x_pixel'] = coor['imagecol'].tolist()
        adata.obs['y_pixel'] = coor['imagerow'].tolist()
        label = pd.read_csv(os.path.join('../data', dataset,  'label_truth.txt'), sep='\t', header=None,
                            index_col=0)
        label.columns = ['over','Ground Truth']
        labels = pd.factorize(label["Ground Truth"].astype("category"))[0]
        adata.obs['Ground Truth'] = label.iloc[:,1]


    #Expression data preprocessing
    adata.var_names_make_unique()
    prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
    #Normalize and take log for UMI
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    pca = PCA(n_components = npca)
    if issparse(adata.X):
        pca.fit(adata.X.A)  #fit(X)，表示用数据X来训练PCA模型。  函数返回值：调用fit方法的对象本身。比如pca.fit(X)，表示用X对pca这个对象进行训练。
        embed=pca.transform(adata.X.A)  #用X来训练PCA模型，同时返回降维后的数据,embed就是降维后的数据。
    else:
        pca.fit(adata.X)
        embed=pca.transform(adata.X)
    features = torch.FloatTensor(np.array(embed))
    return adata,features,labels





def load_graph(dataset,sicle,l):

    if dataset =='DLPFC':
        adj_2path = os.path.join('../data',dataset,sicle,sicle+'_Cosine20_adj.csv')
        input_dir = os.path.join('../data',dataset,sicle)
        adata = sc.read_visium(path=input_dir, count_file=sicle+'_filtered_feature_bc_matrix.h5')
        x_array = adata.obs["array_row"]
        y_array = adata.obs["array_col"]
    elif dataset == 'Human breast cancer':
        adj_2path = os.path.join('../data', dataset, 'Human_breast_cancer_Cosine20_adj.csv')
        input_dir = os.path.join('../data', dataset)
        adata = sc.read_visium(path=input_dir, count_file='V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5')
        x_array = adata.obs["array_row"]
        y_array = adata.obs["array_col"]
    else:
        adj_2path = os.path.join('../data',dataset,dataset+'_Cosine20_adj.csv')
        pos = pd.read_csv('../data/'+dataset+'/pos.csv',sep=',', header=None)
        x_array = pos.iloc[:,0]
        y_array = pos.iloc[:,1]
        print("......")


    adj=calculate_adj_matrix(x_array,y_array)
    adj_1 = np.exp(-1*(adj**2)/(2*(l**2)))
    adj_1 = sp.coo_matrix(adj_1)
    adj_1 = normalize(adj_1 + sp.eye(adj_1.shape[0]))
    adj_1 = sparse_mx_to_torch_sparse_tensor(adj_1)
    adj_2 = np.loadtxt(adj_2path, delimiter=',')
    adj_2 = sp.coo_matrix(adj_2)
    adj_2 = normalize(adj_2 + sp.eye(adj_2.shape[0]))
    adj_2 = sparse_mx_to_torch_sparse_tensor(adj_2)
    return adj_1,adj_2





