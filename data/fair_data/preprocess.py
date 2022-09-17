
import os
from xml.dom.pulldom import default_bufsize
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from collections import defaultdict
import pickle
def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="credit/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

#    # Normalize MaxBillAmountOverLast6Months
#    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
#
#    # Normalize MaxPaymentAmountOverLast6Months
#    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
#
#    # Normalize MostRecentBillAmount
#    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
#
#    # Normalize MostRecentPaymentAmount
#    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
#
#    # Normalize TotalMonthsOverdue
#    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()

    # build relationship
    # if os.path.exists(f'{path}/{dataset}_edges.txt'):
    #     edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    # else:
    #     edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
    #     np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)
    sens_idx = 1
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    features = np.array(features.todense())
    norm_features = np.array(feature_norm(features))
    norm_features[:, sens_idx] = np.array(features[:, sens_idx])
    features = np.array(norm_features)
    edges_unordered = build_relationship(features, thresh=0.95)
    he_index = defaultdict(list)
    for edge in edges_unordered:
        i, j = edge[0], edge[1]
        he_index[i].append(j)
    he_id = 0

    he_index_all = {}
    for key, value in he_index.items():
        he_index_all[he_id] = []
        he_index_all[he_id].extend(value)
        he_id+=1
    labels = idx_features_labels[predict_attr].values
    with open(path+"/features.pickle","wb") as f:
        pickle.dump(features,f)
    with open(path+"/labels.pickle","wb") as f:
        pickle.dump(labels,f)
    with open(path+"/hypergraph.pickle","wb") as f:
        pickle.dump(he_index_all,f)


    # features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    # labels = idx_features_labels[predict_attr].values
    # idx = np.arange(features.shape[0])
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=int).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)

    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])

    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(labels)

    # import random
    # random.seed(20)
    # label_idx_0 = np.where(labels==0)[0]
    # label_idx_1 = np.where(labels==1)[0]
    # random.shuffle(label_idx_0)
    # random.shuffle(label_idx_1)

    # idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    # idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    # idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    # sens = idx_features_labels[sens_attr].values.astype(int)
    # sens = torch.FloatTensor(sens)
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
    
    # return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="bail/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    
    # # Normalize School
    # idx_features_labels['SCHOOL'] = 2*(idx_features_labels['SCHOOL']-idx_features_labels['SCHOOL'].min()).div(idx_features_labels['SCHOOL'].max() - idx_features_labels['SCHOOL'].min()) - 1

    # # Normalize RULE
    # idx_features_labels['RULE'] = 2*(idx_features_labels['RULE']-idx_features_labels['RULE'].min()).div(idx_features_labels['RULE'].max() - idx_features_labels['RULE'].min()) - 1

    # # Normalize AGE
    # idx_features_labels['AGE'] = 2*(idx_features_labels['AGE']-idx_features_labels['AGE'].min()).div(idx_features_labels['AGE'].max() - idx_features_labels['AGE'].min()) - 1

    # # Normalize TSERVD
    # idx_features_labels['TSERVD'] = 2*(idx_features_labels['TSERVD']-idx_features_labels['TSERVD'].min()).div(idx_features_labels['TSERVD'].max() - idx_features_labels['TSERVD'].min()) - 1

    # # Normalize FOLLOW
    # idx_features_labels['FOLLOW'] = 2*(idx_features_labels['FOLLOW']-idx_features_labels['FOLLOW'].min()).div(idx_features_labels['FOLLOW'].max() - idx_features_labels['FOLLOW'].min()) - 1

    # # Normalize TIME
    # idx_features_labels['TIME'] = 2*(idx_features_labels['TIME']-idx_features_labels['TIME'].min()).div(idx_features_labels['TIME'].max() - idx_features_labels['TIME'].min()) - 1

    # build relationship
    # if os.path.exists(f'{path}/{dataset}_edges.txt'):
    #     edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    # else:
    #     edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
    #     np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)
    
    sens_idx = 0
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    features = np.array(features.todense())
    norm_features = np.array(feature_norm(features))
    norm_features[:, sens_idx] = np.array(features[:, sens_idx])
    features = np.array(norm_features)
    edges_unordered = build_relationship(features, thresh=0.95)
    he_index = defaultdict(list)
    for edge in edges_unordered:
        i, j = edge[0], edge[1]
        he_index[i].append(j)
    he_id = 0
    he_index_all = {}
    for key, value in he_index.items():
        he_index_all[he_id] = []
        he_index_all[he_id].extend(value)
        he_id+=1
    labels = idx_features_labels[predict_attr].values
    with open(path+"/features.pickle","wb") as f:
        pickle.dump(features,f)
    with open(path+"/labels.pickle","wb") as f:
        pickle.dump(labels,f)
    with open(path+"/hypergraph.pickle","wb") as f:
        pickle.dump(he_index_all,f)

    # idx = np.arange(features.shape[0])
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=int).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)

    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # # features = normalize(features)
    # adj = adj + sp.eye(adj.shape[0])

    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(labels)

    # import random
    # random.seed(20)
    # label_idx_0 = np.where(labels==0)[0]
    # label_idx_1 = np.where(labels==1)[0]
    # random.shuffle(label_idx_0)
    # random.shuffle(label_idx_1)
    # idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    # idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    # idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    # sens = idx_features_labels[sens_attr].values.astype(int)
    # sens = torch.FloatTensor(sens)
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
    
    # return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="german/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    print(idx_features_labels)
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

#    for i in range(idx_features_labels['PurposeOfLoan'].unique().shape[0]):
#        val = idx_features_labels['PurposeOfLoan'].unique()[i]
#        idx_features_labels['PurposeOfLoan'][idx_features_labels['PurposeOfLoan'] == val] = i

#    # Normalize LoanAmount
#    idx_features_labels['LoanAmount'] = 2*(idx_features_labels['LoanAmount']-idx_features_labels['LoanAmount'].min()).div(idx_features_labels['LoanAmount'].max() - idx_features_labels['LoanAmount'].min()) - 1
#
#    # Normalize Age
#    idx_features_labels['Age'] = 2*(idx_features_labels['Age']-idx_features_labels['Age'].min()).div(idx_features_labels['Age'].max() - idx_features_labels['Age'].min()) - 1
#
#    # Normalize LoanDuration
#    idx_features_labels['LoanDuration'] = 2*(idx_features_labels['LoanDuration']-idx_features_labels['LoanDuration'].min()).div(idx_features_labels['LoanDuration'].max() - idx_features_labels['LoanDuration'].min()) - 1
#
    # build relationship
    # if os.path.exists(f'{path}/{dataset}_edges.txt'):
    #     edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    # else:
    #     edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
    #     np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)
    sens_idx = 0
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    features = np.array(features.todense())
    
    norm_features = np.array(feature_norm(features))
    norm_features[:, sens_idx] = np.array(features[:, sens_idx])
    features = np.array(norm_features)
    edges_unordered = build_relationship(features, thresh=0.95)
    he_index = defaultdict(list)
    for edge in edges_unordered:
        i, j = edge[0], edge[1]
        he_index[i].append(j)
    he_id = 0
    he_index_all = {}
    for key, value in he_index.items():
        he_index_all[he_id] = []
        he_index_all[he_id].extend(value)
        he_id+=1
    # print(he_index_all[1])
    
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0
    with open(path+"/features.pickle","wb") as f:
        pickle.dump(features,f)
    with open(path+"/labels.pickle","wb") as f:
        pickle.dump(labels,f)
    with open(path+"/hypergraph.pickle","wb") as f:
        pickle.dump(he_index_all,f)

    # features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    # labels = idx_features_labels[predict_attr].values
    # labels[labels == -1] = 0

    # idx = np.arange(features.shape[0])
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=int).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # adj = adj + sp.eye(adj.shape[0])

    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(labels)

    # import random
    # random.seed(20)
    # label_idx_0 = np.where(labels==0)[0]
    # label_idx_1 = np.where(labels==1)[0]
    # random.shuffle(label_idx_0)
    # random.shuffle(label_idx_1)

    # idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    # idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    # idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    # sens = idx_features_labels[sens_attr].values.astype(int)
    # sens = torch.FloatTensor(sens)
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
   
    # return adj, features, labels, idx_train, idx_val, idx_test, sens

def build_relationship(x, thresh=0.25):
    # df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    # df_euclid = df_euclid.to_numpy()
    df_euclid = 1 / (1 + distance_matrix(x, x))
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-6]
        neig_id = np.where(df_euclid[ind, :] >= max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map =  np.array(idx_map)
    return idx_map

def feature_norm(features):
    min_values = features.min(axis=0)
    max_values = features.max(axis=0)
    return 2*(features - min_values)/(max_values-min_values) - 1

dataset_list = ["bail",'credit','german']
dataset_list = ['german']
for dataset in dataset_list:
    if dataset == 'credit':
        sens_attr = "Age"  # column number after feature process is 1
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        label_number = 6000
        path_credit = "./credit"
        load_credit(dataset, sens_attr,
                                                                                predict_attr, path=path_credit,
                                                                                label_number=label_number
                                                                                )

    # Load german dataset
    elif dataset == 'german':
        sens_attr = "Gender"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "GoodCustomer"
        label_number = 100
        path_german = "./german"
        load_german(dataset, sens_attr,
                                                                                predict_attr, path=path_german,
                                                                                label_number=label_number,
                                                                                )
    # Load bail dataset
    elif dataset == 'bail':
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        path_bail = "./bail"
        load_bail(dataset, sens_attr, 
                                                                                predict_attr, path=path_bail,
                                                                                label_number=label_number,
                                                                                )