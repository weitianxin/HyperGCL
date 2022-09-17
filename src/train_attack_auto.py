#!/usr/bin/env python
# coding: utf-8

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
from generator import *
from sampling import negative_sampling
def fix_seed(seed=37):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_method(args, data):
    #     Currently we don't set hyperparameters w.r.t. different dataset
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)
    
    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(args,data.norm)
        else:
            model = SetGNN(args)

#     elif args.method == 'SetGPRGNN':
#         model = SetGPRGNN(args)

    elif args.method == 'CEGCN':
        model = CEGCN(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'CEGAT':
        model = CEGAT(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      heads=args.heads,
                      output_heads=args.output_heads,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'HyperGCN':
        #         ipdb.set_trace()
        He_dict = get_HyperGCN_He_dict(data)
        model = HyperGCN(V=data.x.shape[0],
                         E=He_dict,
                         X=data.x,
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args
                         )

    elif args.method == 'HGNN':
        # model = HGNN(in_ch=args.num_features,
        #              n_class=args.num_classes,
        #              n_hid=args.MLP_hidden,
        #              dropout=args.dropout)
        model = HCHA(args)

    elif args.method == 'HNHN':
        model = HNHN(args)

    elif args.method == 'HCHA':
        model = HCHA(args)

    elif args.method == 'MLP':
        model = MLP_model(args)
    elif args.method == 'UniGCNII':
            if args.cuda in [0, 1, 2, 3]:
                device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            (row, col), value = torch_sparse.from_scipy(data.edge_index)
            V, E = row, col
            V, E = V.to(device), E.to(device)
            model = UniGCNII(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes, nlayer=args.All_num_layers, nhead=args.heads,
                             V=V, E=E)
    #     Below we can add different model, such as HyperGCN and so on
    return model


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            best_epoch = []
            for r in result:
                index = np.argmax(r[:, 1])
                best_epoch.append(index)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            print("best epoch:", best_epoch)
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
#             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])


@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

#     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out


@torch.no_grad()
def evaluate_finetune(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model.forward_finetune(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

#     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

#     ipdb.set_trace()
#     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Main part of the training ---
# # Part 0: Parse arguments

def permute_edges(data, aug_ratio, permute_self_edge):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    # if not permute_self_edge:
    permute_num = int((edge_num-node_num) * aug_ratio)
    edge_index = data.edge_index.cpu().numpy()
    if args.add_e:
        idx_add_1 = np.random.choice(node_num, permute_num)
        idx_add_2 = np.random.choice(int(data.num_hyperedges[0].item()), permute_num)
        idx_add = np.stack((idx_add_1, idx_add_2), axis=0)
    # if permute_self_edge:
        # edge_after_remove = edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)]
    #     edge2remove_index = np.where(edge_index[1] < data.num_hyperedges[0].item())[0]
    #     edge2keep_index = np.where(edge_index[1] >= data.num_hyperedges[0].item())[0]
    #     edge_remove_index = np.random.choice(edge2remove_index, permute_num, replace=False)
    #     edge_keep_index = list(set(list(range(edge_num)))-set(edge_remove_index))
    #     edge_after_remove = edge_index[:, edge_keep_index]
    # else:
    edge2remove_index = np.where(edge_index[1] < data.num_hyperedges[0].item())[0]
    edge2keep_index = np.where(edge_index[1] >= data.num_hyperedges[0].item())[0]
    edge_keep_index = np.random.choice(edge2remove_index, (edge_num-node_num)-permute_num, replace=False)
    edge_after_remove1 = edge_index[:, edge_keep_index]
    edge_after_remove2 = edge_index[:, edge2keep_index]
    if args.add_e:
        edge_index = np.concatenate((edge_after_remove1, edge_after_remove2, idx_add), axis=1)
    else:
        # edge_index = edge_after_remove
        edge_index = np.concatenate((edge_after_remove1, edge_after_remove2),axis=1)
    data.edge_index = torch.tensor(edge_index)
    return data


def permute_hyperedges(data, aug_ratio):
    
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    hyperedge_num = int(data.num_hyperedges[0].item())
    permute_num = int(hyperedge_num * aug_ratio)
    index = defaultdict(list)
    edge_index = data.edge_index.cpu().numpy()
    # time1 = time.time()
    # for i, he in enumerate(edge_index[1]):
    #     index[he].append(i)
    # time2 = time.time()

    # edge2keep_index = np.where(edge_index[1] >= data.num_hyperedges[0].item())[0]
    # edge_keep_index = np.random.choice(hyperedge_num, hyperedge_num-permute_num, replace=False)
    # edge_keep_index_all = []
    # for keep_index in edge_keep_index:
    #     edge_keep_index_all.extend(he_index[keep_index])
    # edge_keep_index_all = np.concatenate((edge_keep_index_all, edge2keep_index))
    # edge_after_remove = edge_index[:, edge_keep_index_all]
    # edge_index = edge_after_remove
    edge_remove_index = np.random.choice(hyperedge_num, permute_num, replace=False)
    edge_remove_index_dict={ind:i for i,ind in enumerate(edge_remove_index)}
    # edge_remove_index_all = []
    # for remove_index in edge_remove_index:
    #     edge_remove_index_all.extend(he_index[remove_index])
    edge_remove_index_all = [i for i, he in enumerate(edge_index[1]) if he in edge_remove_index_dict]
    # print(len(edge_remove_index_all), edge_num, len(edge_remove_index), aug_ratio, hyperedge_num)
    edge_keep_index = list(set(list(range(edge_num)))-set(edge_remove_index_all))
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = edge_after_remove

    data.edge_index = torch.tensor(edge_index)
    return data


def adapt(data, aug_ratio, aug):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    hyperedge_num = int(data.num_hyperedges[0].item())
    permute_num = int(hyperedge_num * aug_ratio)
    index = defaultdict(list)
    edge_index = data.edge_index.cpu().numpy()
    for i, he in enumerate(edge_index[1]):
        index[he].append(i)
    # edge
    drop_weights = degree_drop_weights(data.edge_index, hyperedge_num)
    edge_index_1 = drop_edge_weighted(data.edge_index, drop_weights, p=aug_ratio, threshold=0.7, h=hyperedge_num, index=index)
    
    # feature
    edge_index_ = data.edge_index
    node_deg = degree(edge_index_[0])
    feature_weights = feature_drop_weights(data.x, node_c=node_deg)
    x_1 = drop_feature_weighted(data.x, feature_weights, aug_ratio, threshold=0.7)
    if aug=="adapt_edge":
        data.edge_index = edge_index_1
    elif aug=="adapt_feat":
        data.x = x_1
    else:
        data.edge_index = edge_index_1
        data.x = x_1
    return data

def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x

def degree_drop_weights(edge_index, h):
    edge_index_ = edge_index
    deg = degree(edge_index_[1])[:h]
    # deg_col = deg[edge_index[1]].to(torch.float32)
    deg_col = deg
    s_col = torch.log(deg_col)
    # weights = (s_col.max() - s_col+1e-9) / (s_col.max() - s_col.mean()+1e-9)
    weights = (s_col - s_col.min()+1e-9) / (s_col.mean() - s_col.min()+1e-9)
    return weights

def feature_drop_weights(x, node_c):
    # x = x.to(torch.bool).to(torch.float32)
    x = torch.abs(x).to(torch.float32)
    # 100 x 2012 mat 2012-> 100
    w = x.t() @ node_c
    w = w.log()
    # s = (w.max() - w) / (w.max() - w.mean())
    s = (w - w.min()) / (w.mean() - w.min())
    return s

def drop_edge_weighted(edge_index, edge_weights, p: float, h, index, threshold: float = 1.):
    _, edge_num = edge_index.size()
    edge_weights = (edge_weights+1e-9) / (edge_weights.mean()+1e-9) * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    # keep probability
    sel_mask = torch.bernoulli(edge_weights).to(torch.bool)
    edge_remove_index = np.array(list(range(h)))[sel_mask.cpu().numpy()]
    edge_remove_index_all = []
    for remove_index in edge_remove_index:
        edge_remove_index_all.extend(index[remove_index])
    edge_keep_index = list(set(list(range(edge_num)))-set(edge_remove_index_all))
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = edge_after_remove
    return edge_index

def mask_nodes(data, aug_ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    zero_v = torch.zeros_like(token)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = token

    return data

def mask_nodes_zero(data, aug_ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = 0

    return data

def mask_nodes_col(data, aug_ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(feat_dim * aug_ratio)

    idx_mask = np.random.choice(feat_dim, mask_num, replace=False)
    data.x[:, idx_mask] = 0

    return data

def drop_nodes(data, aug_ratio):
    
    node_size = int(data.n_x[0].item())
    sub_size = int(node_size*(1-aug_ratio))
    hyperedge_size = int(data.num_hyperedges[0].item())
    sample_nodes = np.random.permutation(node_size)[:sub_size]
    sample_nodes = list(np.sort(sample_nodes))
    edge_index = data.edge_index
    device = edge_index.device
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(sample_nodes, 1, edge_index, relabel_nodes=False, flow='target_to_source')
    sub_nodes, sorted_idx = torch.sort(sub_nodes)
    # relabel
    node_idx = torch.zeros(2*node_size+hyperedge_size, dtype=torch.long, device=device)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)
    sub_edge_index = node_idx[sub_edge_index]
    data.x = data.x[sample_nodes]
    data.edge_index = sub_edge_index

    data.n_x = torch.tensor([sub_size])
    data.num_hyperedges = torch.tensor([sub_nodes.size(0) - 2*sub_size])
    data.norm = 0
    data.totedges = torch.tensor(sub_nodes.size(0) - sub_size)
    data.num_ori_edge = sub_edge_index.shape[1] - sub_size
    return data, set(sub_nodes[:sub_size].cpu().numpy())

def subgraph_aug(data, aug_ratio, start):
    
    n_walkLen = 16
    node_num, _ = data.x.size()
    he_num = data.totedges.item()
    edge_index = data.edge_index

    row, col = edge_index
    # torch.cat([row,col])
    # adj = SparseTensor(row=torch.cat([row,col]), col=torch.cat([col,row]), sparse_sizes=(node_num+he_num, he_num+node_num))
    adj = SparseTensor(row=torch.cat([row,col]), col=torch.cat([col,row]), sparse_sizes=(node_num+he_num, he_num+node_num))
    
    node_idx = adj.random_walk(start.flatten(), n_walkLen).view(-1)
    sub_nodes = node_idx.unique()
    sub_nodes.sort()
    sub_edge_index, _ = subgraph(sub_nodes, edge_index, relabel_nodes=True)
    data.edge_index = sub_edge_index
    cidx = torch.where(sub_nodes >= node_num)[
        0].min()
    data.x = data.x[sub_nodes[:cidx]]
    return data, set(sub_nodes[:cidx].cpu().numpy())

def aug(data, args, start=None):
    data_aug = copy.deepcopy(data)
    if args.aug=="mask":
        data_aug = mask_nodes(data_aug, args.aug_ratio)
    elif args.aug=="edge":
        data_aug = permute_edges(data_aug, args.aug_ratio, args.permute_self_edge)
    elif args.aug=="hyperedge":
        data_aug = permute_hyperedges(data_aug, args.aug_ratio)
    elif args.aug=="mask_col":
        data_aug = mask_nodes_col(data_aug, args.aug_ratio)
    elif args.aug=="mask_zero":
        data_aug = mask_nodes_zero(data_aug, args.aug_ratio)
    elif args.aug=="subgraph":
        data_aug, sample_nodes = subgraph_aug(data_aug, args.aug_ratio, start)
        return data_aug, sample_nodes
    elif args.aug=="drop":
        data_aug, sample_nodes = drop_nodes(data_aug, args.aug_ratio)
        return data_aug, sample_nodes
    elif args.aug=="none":
        return data_aug
    elif "adapt" in args.aug:
        data_aug = adapt(data_aug, args.aug_ratio, args.aug)
    else:
        raise ValueError(f'not supported augmentation')
    return data_aug

def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor, T):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def whole_batched_semi_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, T):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / T)
    indices = torch.arange(0, num_nodes).to(device)
    losses = []
    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        refl_sim = f(sim(z1[mask], z1))  # [B, N]
        between_sim = f(sim(z1[mask], z2))  # [B, N]

        losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                    / (refl_sim.sum(1) + between_sim.sum(1)
                                    - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

def batched_semi_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, T):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / T)
    indices = np.arange(0, num_nodes)
    np.random.shuffle(indices)
    i = 0
    mask = indices[i * batch_size:(i + 1) * batch_size]
    refl_sim = f(sim(z1[mask], z1))  # [B, N]
    between_sim = f(sim(z1[mask], z2))  # [B, N]
    loss = -torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                / (refl_sim.sum(1) + between_sim.sum(1)
                                - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag()))

    return loss

def com_semi_loss(z1: torch.Tensor, z2: torch.Tensor, T, com_nodes1, com_nodes2):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim[com_nodes1,com_nodes2] / (refl_sim.sum(1)[com_nodes1] + between_sim.sum(1)[com_nodes1] - refl_sim.diag()[com_nodes1]))


def contrastive_loss_node(x1, x2, args, com_nodes=None):
    T = args.t
    # if args.dname in ["yelp", "coauthor_dblp", "walmart-trips-100"]:
    #     batch_size=1024
    # else:
    #     batch_size = None
    batch_size = None
    if com_nodes is None:
        if batch_size is None:
            l1 = semi_loss(x1, x2, T)
            l2 = semi_loss(x2, x1, T)
        else:
            l1 = batched_semi_loss(x1, x2, batch_size, T)
            l2 = batched_semi_loss(x2, x1, batch_size, T)
    else:
        l1 = com_semi_loss(x1, x2, T, com_nodes[0], com_nodes[1])
        l2 = com_semi_loss(x2, x1, T, com_nodes[1], com_nodes[0])
    ret = (l1 + l2) * 0.5
    ret = ret.mean()
    
    return ret

def sim_d(z1: torch.Tensor, z2: torch.Tensor):
        # z1 = F.normalize(z1)
        # z2 = F.normalize(z2)
        return torch.sqrt(torch.sum(torch.pow(z1-z2,2),1))

def calculate_distance(z1: torch.Tensor, z2: torch.Tensor):
    num_nodes = z1.size(0)
    refl_sim = 0
    for i in range(num_nodes):
        refl_sim += (torch.sum(sim_d(z1[i:i+1], z1)) - torch.squeeze(sim_d(z1[i:i+1], z1[i:i+1])))/(num_nodes-1)
    refl_sim = refl_sim/(num_nodes)
    between_sim = torch.sum(sim_d(z1, z2))/num_nodes
    print(refl_sim, between_sim)

def remove_attack(data, args):
    aug_ratio = args.a_ratio
    _, edge_num = data.edge_index.size()
    # if not permute_self_edge:
    permute_num = int(edge_num * aug_ratio)
    edge_index = data.edge_index.cpu().numpy()
    edge2remove_index = list(range(edge_num))
    edge_keep_index = np.random.choice(edge2remove_index, edge_num-permute_num, replace=False)
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = edge_after_remove
    data.edge_index = torch.tensor(edge_index)
    return data

def create_hypersubgraph(data, args):
    
    sub_size = args.sub_size
    node_size = int(data.n_x[0].item())
    hyperedge_size = int(data.num_hyperedges[0].item())
    sample_nodes = np.random.permutation(node_size)[:sub_size]
    sample_nodes = list(np.sort(sample_nodes))
    edge_index = data.edge_index
    device = edge_index.device
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(sample_nodes, 1, edge_index, relabel_nodes=False, flow='target_to_source')
    sub_nodes, sorted_idx = torch.sort(sub_nodes)
    # relabel
    node_idx = torch.zeros(2*node_size+hyperedge_size, dtype=torch.long, device=device)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)
    sub_edge_index = node_idx[sub_edge_index]
    x = data.x[sample_nodes]
    data_sub = Data(x=x, edge_index=sub_edge_index)
    data_sub.n_x = torch.tensor([sub_size])
    data_sub.num_hyperedges = torch.tensor([sub_nodes.size(0) - 2*sub_size])
    data_sub.norm = 0
    data_sub.totedges = torch.tensor(sub_nodes.size(0) - sub_size)
    data_sub.num_ori_edge = sub_edge_index.shape[1] - sub_size
    return data_sub

def attack_preprocess(attack_adj):
    if attack_adj.shape[0]==2:
        attack_adj = np.transpose(attack_adj)
    n_edge = attack_adj.shape[0]
    n_nodes = data.x.shape[0]
    n_hes = data.num_hyperedges[0].item()
    ano=0
    new_he = defaultdict(list)
    for value in attack_adj:
        i, j = value[0], value[1]
        if i<n_nodes and j<n_nodes:
            ano+=1
            new_he[i].append(j)
    remove_index1 = set(np.where(attack_adj[:,0]<n_nodes)[0])&set(np.where(attack_adj[:,1]<n_nodes)[0])
    remove_index2 = set(np.where(attack_adj[:,0]>=n_nodes)[0])&set(np.where(attack_adj[:,1]>=n_nodes)[0])
    remove_index = remove_index1|remove_index2
    print(remove_index2)
    keep_index = list(set(range(n_edge))-remove_index)
    attack_keep_adj = attack_adj[keep_index]
    node_list = list(attack_keep_adj[:,0])
    edge_list = list(attack_keep_adj[:,1])
    new_he_index = n_nodes+n_hes
    for key, values in new_he.items():
        if len(values)>1:
            values.append(key)
            cur_size = len(values)
            node_list += list(values)
            edge_list += [new_he_index] * cur_size
            node_list += [new_he_index] * cur_size
            edge_list += list(values)
            new_he_index += 1
    edge_index = np.array([ node_list,
                            edge_list], dtype = np.int)
    edge_index = torch.LongTensor(edge_index)
    return edge_index, new_he_index-n_nodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
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
    parser.add_argument('--attack', type=str, default = "minmax", help="remove|minmax|net")
    parser.add_argument('--a_ratio', type=float, default = 0.1)
    parser.add_argument('--seed', type=int, default = 123)
    parser.add_argument('--hard', type=int, default = 1)
    parser.add_argument('--deg', type=int, default = 0)
    parser.add_argument('--g_lr', type=float, default = 1e-3)
    parser.add_argument('--g_l', type=float, default = 1)
    parser.add_argument('--d_l', type=float, default = 0)
    parser.add_argument('--step', type=int, default = 1)
    parser.add_argument('--multi', action='store_true', default = False)
    parser.add_argument('--easy', action='store_true', default = False)
    parser.add_argument('--aug_two', type=int, default = 0)
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
    fix_seed(args.seed)
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
        attack_root = "../data/attack_data"
        if args.attack=="remove":
            data = remove_attack(data, args)
        else:
            attack_file_name = "{}_{}_adj_0.1_0.1.npz".format(dname, args.attack)
            attack_adj = sp.load_npz(osp.join(attack_root,dname,attack_file_name)).toarray()
            attack_edge_index, n_hes = attack_preprocess(attack_adj)
            data.edge_index = attack_edge_index
            data.num_hyperedges = torch.tensor(
                [n_hes])
    # ipdb.set_trace()
    #     Preprocessing
    # if args.method in ['SetGNN', 'SetGPRGNN', 'SetGNN-DeepSet']:
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        if args.exclude_self:
            data = expand_edge_index(data)
        #     Compute deg normalization: option in ['all_one','deg_half_sym'] (use args.normtype)
        # data.norm = torch.ones_like(data.edge_index[0])
        data = norm_contruction(data, option=args.normtype)
    elif args.method in ['CEGCN', 'CEGAT']:
        data = ExtractV2E(data)
        data = ConstructV2V(data)
        data = norm_contruction(data, TYPE='V2V')
    
    elif args.method in ['HyperGCN']:
        data = ExtractV2E(data)
    #     ipdb.set_trace()
    #   Feature normalization, default option in HyperGCN
        # X = data.x
        # X = sp.csr_matrix(utils.normalise(np.array(X)), dtype=np.float32)
        # X = torch.FloatTensor(np.array(X.todense()))
        # data.x = X
    
    # elif args.method in ['HGNN']:
    #     data = ExtractV2E(data)
    #     if args.add_self_loop:
    #         data = Add_Self_Loops(data)
    #     data = ConstructH(data)
    #     data = generate_G_from_H(data)
    
    elif args.method in ['HNHN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        H = ConstructH_HNHN(data)
        data = generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()
    
    elif args.method in ['HCHA', 'HGNN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
    #    Make the first he_id to be 0
        data.edge_index[1] -= data.edge_index[1].min()
        
    elif args.method in ['UniGCNII']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data = ConstructH(data)
        data.edge_index = sp.csr_matrix(data.edge_index)
        # Compute degV and degE
        if args.cuda in [0,1,2,3]:
            device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        (row, col), value = torch_sparse.from_scipy(data.edge_index)
        V, E = row, col
        V, E = V.to(device), E.to(device)

        degV = torch.from_numpy(data.edge_index.sum(1)).view(-1, 1).float().to(device)
        from torch_scatter import scatter
        degE = scatter(degV[V], E, dim=0, reduce='mean')
        degE = degE.pow(-0.5)
        degV = degV.pow(-0.5)
        degV[torch.isinf(degV)] = 1
        args.UniGNN_degV = degV
        args.UniGNN_degE = degE
    
        V, E = V.cpu(), E.cpu()
        del V
        del E
    
    #     Get splits
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)
    
    
    # # Part 2: Load model
    
    
    model = parse_method(args, data)
    # put things to device
    if args.cuda in [0, 1, 2, 3]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    # load attack graph
    model = model.to(device)
    data = data.to(device)
    data_pre = copy.deepcopy(data)
    if args.method == 'UniGCNII':
        args.UniGNN_degV = args.UniGNN_degV.to(device)
        args.UniGNN_degE = args.UniGNN_degE.to(device)
    
    num_params = count_parameters(model)
    
    
    # # Part 3: Main. Training + Evaluation
    
    
    logger = Logger(args.runs, args)
    
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    
    model.train()
    # print('MODEL:', model)
    ### Training loop ###
    he_index = defaultdict(list)
    edge_index = data.edge_index.cpu().numpy()
    for i, he in enumerate(edge_index[1]):
        he_index[he].append(i)
    runtime_list = []
    for run in tqdm(range(args.runs)):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        model.reset_parameters()
        if args.method == 'UniGCNII':
            optimizer = torch.optim.Adam([
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        else:
            # if args.p_lr:
            #     optimizer = torch.optim.Adam(model.linear.parameters(), lr=args.lr, weight_decay=args.wd)
            # else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    #     This is for HNHN only
    #     if args.method == 'HNHN':
    #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100, gamma=0.51)
        # pretrain
        #     best_val = float('-inf')
        #     for epoch in range(args.epochs):
        #         #         Training part
        #         model.train()
        #         optimizer.zero_grad()
        #         out = model.forward_finetune(data)
        #         out = F.log_softmax(out, dim=1)
        #         loss = criterion(out[train_idx], data.y[train_idx])
        #         loss.backward()
        #         optimizer.step()
        # #         if args.method == 'HNHN':
        # #             scheduler.step()
        # #         Evaluation part
        #         result = evaluate_finetune(model, data, split_idx, eval_func)
        #         logger.add_result(run, result[:3])
        
        #         if epoch % args.display_step == 0 and args.display_step > 0:
        #             print(f'Epoch: {epoch:02d}, '
        #                 f'Train Loss: {loss:.4f}, '
        #                 f'Valid Loss: {result[4]:.4f}, '
        #                 f'Test  Loss: {result[5]:.4f}, '
        #                 f'Train Acc: {100 * result[0]:.2f}%, '
        #                 f'Valid Acc: {100 * result[1]:.2f}%, '
        #                 f'Test  Acc: {100 * result[2]:.2f}%')
        best_val = float('-inf')
        encoder, decoder = vhgae_encoder(args).to(device), vhgae_decoder(args).to(device)
        encoder.reset_parameters()
        view_generator = vhgae(encoder, decoder, args).to(device)
        optimizer_g = torch.optim.Adam(view_generator.parameters(), lr=args.g_lr, weight_decay=0)
        for epoch in range(args.epochs):
            if data_pre.n_x[0].item()<=args.sub_size:
                data_sub = data_pre
            else:
                data_sub = create_hypersubgraph(data_pre, args)
            # generator update
            view_generator.train()
            model.train()
            data_aug1 = data_sub.to(device)
            for sth in range(args.step):
                optimizer_g.zero_grad()
                out = model.forward_cl(data_aug1)
                data_2 = copy.deepcopy(data_sub).to(device)
                if not args.deg:
                    data_2.edge_index_neg = negative_sampling(data_2.edge_index, num_nodes=[int(data_2.n_x[0].item()), int(data_2.totedges.item())])
                loss_vhgae, data_aug2, aug_weight, drop = view_generator.generate(data_2)
                if args.method=="AllSetTransformer":
                    # mask_value = -1e9*torch.ones(aug_weight.shape).to(device)
                    # aug_weight_attn = torch.unsqueeze((1-aug_weight)*mask_value, 1)
                    aug_weight_attn = torch.unsqueeze(aug_weight, 1)
                else:
                    aug_weight_attn = aug_weight
                out_aug = model.forward_cl(copy.deepcopy(data_sub).to(device), aug_weight_attn)
                loss_cl = contrastive_loss_node(out, out_aug, args)
                # print(sth, drop)
                if epoch==0:
                    g_l=args.g_l
                # else:
                #     if epoch%50==0:
                #         g_l=1.1*g_l
                if args.deg:
                    # loss_generator = torch.square(drop-args.aug_ratio)-args.g_l*loss_cl
                    # loss_generator = torch.abs(drop-args.aug_ratio)-args.g_l*loss_cl
                    loss_generator = torch.abs(drop-0.3)-g_l*loss_cl
                    # loss_generator = drop
                else:
                    loss_generator = loss_vhgae-g_l*loss_cl+args.d_l*torch.abs(drop-0.3)
                # out = torch.autograd.grad(loss_cl, view_generator.encoder.V2EConvs[0].f_enc.parameters())
                loss_generator.backward()
                optimizer_g.step()
            #         Training part
            model.train()
            optimizer.zero_grad()
            # cl loss
            
            cidx = data_sub.edge_index[1].min()
            data_sub.edge_index[1] -= cidx
            data_sub = data_sub.to(device)
            data_aug1 = aug(data_sub, args).to(device)
            # data_aug1 = data_sub
            model.train()
            view_generator.eval()
            # out = model.forward_cl(data_sub)
            if args.aug_two:
                data_1 = copy.deepcopy(data_sub).to(device)
                with torch.no_grad():
                    _, data_aug1, aug_weight1, drop = view_generator.generate_only(data_1)
                if args.method=="AllSetTransformer":
                    aug_weight_attn = torch.unsqueeze(aug_weight1, 1)
                else:
                    aug_weight_attn = aug_weight1
                out = model.forward_cl(copy.deepcopy(data_sub).to(device), aug_weight_attn)
            else:
                out = model.forward_cl(copy.deepcopy(data_aug1))
            data_2 = copy.deepcopy(data_sub).to(device)
            with torch.no_grad():
                _, data_aug2, aug_weight, drop = view_generator.generate_only(data_2)
            # print(drop)
            if args.method=="AllSetTransformer":
                # mask_value = -1e9*torch.ones(aug_weight.shape).to(device)
                # aug_weight_attn = torch.unsqueeze((1-aug_weight)*mask_value, 1)
                aug_weight_attn = torch.unsqueeze(aug_weight, 1)
            else:
                aug_weight_attn = aug_weight
            
            out_aug = model.forward_cl(copy.deepcopy(data_sub).to(device), aug_weight_attn)
            loss_cl = contrastive_loss_node(out, out_aug, args)
            # sup loss
            if args.linear:
                out = model.forward_finetune(data)
            else:
                out = model(data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], data.y[train_idx])
            loss += args.m_l*loss_cl
            # if epoch==10:
            #     print(out_aug, data_aug1.edge_index[0][:100])
            #     print()
            #     # print(list(model.named_parameters()))
            #     exit()
            loss.backward()
            optimizer.step()
    #         if args.method == 'HNHN':
    #             scheduler.step()
    #         Evaluation part
            if args.linear:
                result = evaluate_finetune(model, data, split_idx, eval_func)
            else:
                result = evaluate(model, data, split_idx, eval_func)
            logger.add_result(run, result[:3])
    
            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                    f'Train Loss: {loss:.4f}, '
                    f'Valid Loss: {result[4]:.4f}, '
                    f'Test  Loss: {result[5]:.4f}, '
                    f'Train Acc: {100 * result[0]:.2f}%, '
                    f'Valid Acc: {100 * result[1]:.2f}%, '
                    f'Test  Acc: {100 * result[2]:.2f}%')

        end_time = time.time()
        runtime_list.append(end_time - start_time)
    
        # logger.print_statistics(run)
    
    ### Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

    best_val, best_test = logger.print_statistics()
    res_root = 'hyperparameter_tunning'
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}.csv'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj:
        if args.m_l:
            cur_line = f'attack_{args.attack}_auto__{args.a_ratio}_{args.method}_{args.m_l}_{args.lr}_{args.wd}_{args.sub_size}_{args.heads}_aug_{args.aug}_ratio_{str(args.aug_ratio)}_t_{str(args.t)}_plr_{str(args.p_lr)}_pepoch_{str(args.p_epochs)}_player_{str(args.p_layer)}_phidden_{str(args.p_hidden)}_drop_{str(args.dropout)}_train_{str(args.train_prop)}'
            if args.add_e:
                cur_line+="_add_e"
        else:
            cur_line = f'attack_{args.attack}_{args.a_ratio}_{args.method}_{args.lr}_{args.wd}_{args.heads}_{str(args.dropout)}_train_{str(args.train_prop)}'
        cur_line += f',{best_val.mean():.3f} ± {best_val.std():.3f}'
        cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}'
        cur_line += f',{num_params}, {avg_time:.2f}s, {std_time:.2f}s' 
        cur_line += f',{avg_time//60}min{(avg_time % 60):.2f}s'
        cur_line += f'\n'
        write_obj.write(cur_line)

    all_args_file = f'{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    print('All done! Exit python code')
    quit()
