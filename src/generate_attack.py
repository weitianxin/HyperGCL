import os
import time
# import math
import torch
# import pickle
import argparse
import random
import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

from layers import *
from models import *
from preprocessing import *
from collections import defaultdict
from torch_sparse import SparseTensor, coalesce
from convert_datasets_to_pygDataset import dataset_Hypergraph
from torch_geometric.utils import dropout_adj, degree, to_undirected, k_hop_subgraph, subgraph
from torch_geometric.data import Data



from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, MetaApprox, MinMax, PGDAttack
from deeprobust.graph.targeted_attack import Nettack
# from meta import Metattack
from torch_geometric.nn.conv import MessagePassing
from scatter_func import scatter
# from deeprobust.graph.utils import preprocess

def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor, and normalize the input data.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    preprocess_adj : bool
        whether to normalize the adjacency matrix
    preprocess_feature : bool
        whether to normalize the feature matrix
    sparse : bool
       whether to return sparse tensor
    device : str
        'cpu' or 'cuda'
    """

    if preprocess_adj:
        adj = normalize_adj(adj)

    if preprocess_feature:
        features = normalize_feature(features)

    labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))
        adj = torch.FloatTensor(adj.todense())
    return adj.to(device), features.to(device), labels.to(device)

torch.backends.cudnn.enabled = True

torch.backends.cudnn.benchmark = True
class MeanAggr(MessagePassing):
    def __init__(self):
        super(MeanAggr, self).__init__()

    def forward(self, x, edge_index, aggr='mean'):
        x = self.propagate(edge_index, x=x, aggr=aggr)
        return x

    def message(self, x_j):
        # return norm.view(-1, 1) * x_j
        out = x_j
        return out

    def aggregate(self, inputs, index,
                  dim_size=None, aggr=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)

def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
#     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    num_nodes = data.n_x[0]
    num_hyperedges = data.num_hyperedges[0]
    if not ((data.n_x[0]+data.num_hyperedges[0]-1) == data.edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[
        0].min()  # cidx: [V...|cidx E...]
    edge_index_uni = edge_index[:, :cidx].type(torch.LongTensor)
    return edge_index_uni

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)

def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False

def save_adj(modified_adj, root=r'/tmp/', name='mod_adj'):
        """Save attacked adjacency matrix.

        Parameters
        ----------
        root :
            root directory where the variable should be saved
        name : str
            saved file name

        Returns
        -------
        None.

        """
        assert modified_adj is not None, \
                'modified_adj is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_adj = modified_adj

        if type(modified_adj) is torch.Tensor:
            sparse_adj = to_scipy(modified_adj)
            sp.save_npz(osp.join(root, name), sparse_adj)
        else:
            sp.save_npz(osp.join(root, name), modified_adj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.1)
    parser.add_argument('--valid_prop', type=float, default=0.1)
    parser.add_argument('--dname', default='walmart-trips-100')
    # method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
    parser.add_argument('--method', default='AllSetTransformer')
    parser.add_argument('--epochs', default=500, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=20, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1, 2, 3], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--All_num_layers', default=2, type=int)
    parser.add_argument('--MLP_num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=64,
                        type=int)  # Encoder hidden units
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--display_step', type=int, default=-1)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--deepset_input_norm', default = True)
    parser.add_argument('--GPR', action='store_false')  # skip all but last dec
    # skip all but last dec
    parser.add_argument('--LearnMask', action='store_false')
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=0, type=int)  # Placeholder
    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='1', type=str)
    # whether the he contain self node or not
    parser.add_argument('--exclude_self', action='store_true')
    parser.add_argument('--PMA', action='store_true')
    #     Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', action='store_true')
    #     Args for Attentions: GAT and SetGNN
    parser.add_argument('--heads', default=1, type=int)  # Placeholder
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    #     Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    #     Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    #     Args for UniGNN
    parser.add_argument('--UniGNN_use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--UniGNN_degE', default = 0)
    #     Args for contrastive learning
    parser.add_argument('--t', type=float, default = 0.5)
    parser.add_argument('--p_lr', type=float, default = 0)
    parser.add_argument('--p_epochs', type=int, default = 300)
    parser.add_argument('--aug_ratio', type=float, default = 0.1)
    parser.add_argument('--p_hidden', type=int, default = -1)
    parser.add_argument('--p_layer', type=int, default = -1)
    parser.add_argument('--aug', type=str, default = "edge", help='mask|edge|hyperedge|mask_col|adapt|adapt_feat|adapt_edge')
    parser.add_argument('--add_e', action='store_true', default = False)
    parser.add_argument('--permute_self_edge', action='store_true', default = False)
    parser.add_argument('--linear', action='store_true', default = False)
    parser.add_argument('--sub_size', type=int, default = 16384)
    parser.add_argument('--m_l', type=float, default = 0.1)
    parser.add_argument('--seed', type=int, default = 123)
    parser.add_argument('--ptb_rate', type=float, default = 0.1)
    parser.add_argument('--t_attack', type=int, default = 0)
    parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)
    
    #     Use the line below for .py file
    args = parser.parse_args()
    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
    #     Use the line below for notebook
    # args = parser.parse_args([])
    # args, _ = parser.parse_known_args()
    
    
    # # Part 1: Load data
    
    
    ### Load and preprocess data ###
    existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                        'NTU2012', 'Mushroom',
                        'coauthor_cora', 'coauthor_dblp',
                        'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                        'walmart-trips-100', 'house-committees-100',
                        'cora', 'citeseer', 'pubmed']
    
    synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']
    
    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, 
                    feature_noise=f_noise,
                    p2raw = p2raw)
        else:
            if dname in ['cora', 'citeseer','pubmed']:
                p2raw = '../data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = '../data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['yelp']:
                p2raw = '../data/AllSet_all_raw_data/yelp/'
            else:
                p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw = p2raw)
        data = dataset.data
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        if args.dname in ['yelp', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']:
            #         Shift the y label to start with 0
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
        # if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
        # if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
        data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])
    
    # device = torch.device('cpu')
    #     Get splits
    train_prop = args.train_prop
    labeled_nodes = torch.where(data.y != -1)[0]

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    num_valid = int(n * (train_prop+0.1))

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:num_valid]
    test_indices = perm[num_valid:]

    idx_train = labeled_nodes[train_indices]
    idx_val = labeled_nodes[val_indices]
    idx_test = labeled_nodes[test_indices]
    n_total = data.edge_index[0].max()+1
    n_nodes = data.n_x[0].item()
    adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]),
            (data.edge_index[0], data.edge_index[1])), shape=(n_total, n_total))
    aggr_func = MeanAggr()
    edge_index_uni = ExtractV2E(data)
    edge_index_uni[1] -= edge_index_uni[1].min()
    edge_features = aggr_func(data.x, edge_index_uni)
    features = torch.cat([data.x, edge_features],dim=0)
    labels = data.y
    idx_unlabeled = np.union1d(idx_val, idx_test)
    idx_unlabeled = np.union1d(idx_val, idx_test)
    if args.t_attack==0:
        # Setup Surrogate model
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0, with_relu=False, with_bias=True, device=device).to(device)
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
        # Setup Attack Model
        n_perturbations = int(args.ptb_rate * (adj.sum()//2))
        adj, features, labels = preprocess(adj, features.numpy(), labels, preprocess_adj=False)
        model = MinMax(model=surrogate, nnodes=adj.shape[0], loss_type='CE', device=device).to(device)
        model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbations)
        modified_adj = model.modified_adj # modified_adj is a torch.tensor
        # for i in range(30):

        #     model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
        #             attack_structure=True, attack_features=False, device=device, lambda_=0).to(device)
        #     # Attack
        #     model.attack(features, modified_adj, labels, idx_train, idx_unlabeled, n_perturbations=1, ll_constraint=False)
            
        #     modified_adj = model.modified_adj.cpu()
        #     print(modified_adj)
        #     del model
        #     torch.cuda.empty_cache()
        modified_edge_index = torch.LongTensor(torch.abs(modified_adj.to(device)-adj.to(device)).detach().cpu().nonzero())
        modified_edge_index = modified_edge_index.numpy()
        ano = 0
        for value in modified_edge_index:
            i, j = value[0], value[1]
            if i<n_nodes and j<n_nodes:
                ano+=1
            if i>n_nodes and j>n_nodes:
                ano+=1
        print(ano, modified_edge_index.shape)
        root = "../data/attack_data/{}/".format(args.dname)
        if not osp.isdir(root):
                os.makedirs(root)
        modified_edge_index = torch.tensor(modified_edge_index)
        after_edge_index = torch.LongTensor(modified_adj.cpu().nonzero())
        save_adj(after_edge_index, root=root, name='{}_minmax_adj_{}_{}'.format(args.dname,args.ptb_rate, args.train_prop))
    else:
        idx = list(range(n_nodes))
        np.random.shuffle(idx)
        node_list = idx[:int(args.ptb_rate*len(idx))]
        # Setup Surrogate model
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0, with_relu=False, with_bias=True, device=device).to(device)
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
        modified_adj = adj
        degrees = adj.sum(0).A1
        features = to_scipy(features)
        for target_node in tqdm(node_list):
            # Setup Attack Model
            n_perturbations = int(degrees[target_node])
            n_perturbations = n_perturbations if n_perturbations>0 else 1
            model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
            model = model.to(device)
            model.attack(features, modified_adj, labels, target_node, n_perturbations, verbose=False)
            modified_adj = model.modified_adj
        # for i in range(30):

        #     model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
        #             attack_structure=True, attack_features=False, device=device, lambda_=0).to(device)
        #     # Attack
        #     model.attack(features, modified_adj, labels, idx_train, idx_unlabeled, n_perturbations=1, ll_constraint=False)
            
        #     modified_adj = model.modified_adj.cpu()
        #     print(modified_adj)
        #     del model
        #     torch.cuda.empty_cache()
        modified_edge_index = torch.FloatTensor((modified_adj-adj).todense()).nonzero()
        ano = 0
        for value in modified_edge_index:
            i, j = value[0], value[1]
            if i<n_nodes and j<n_nodes:
                ano+=1
            if i>n_nodes and j>n_nodes:
                ano+=1
        print(ano, modified_edge_index.shape)
        root = "../data/attack_data/{}/".format(args.dname)
        if not osp.isdir(root):
                os.makedirs(root)
        modified_edge_index = torch.tensor(modified_edge_index)
        after_edge_index = torch.LongTensor(modified_adj.nonzero())
        save_adj(after_edge_index, root=root, name='{}_net_adj_{}_{}'.format(args.dname,args.ptb_rate, args.train_prop))
    
    
    
# from deeprobust.graph.data import Dataset, PrePtbDataset, Dpr2Pyg
# data = Dataset(root='/tmp/', name='cora') # load clean graph
# pyg_data = Dpr2Pyg(data) # convert dpr to pyg
# # load perturbed graph
# perturbed_data = PrePtbDataset(root='/tmp/',
#         name='cora',
#         attack_method='meta',
#         ptb_rate=0.05)
# perturbed_adj = perturbed_data.adj
# pyg_data.update_edge_index(perturbed_adj)