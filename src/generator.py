from stringprep import in_table_c11_c12
import torch
import utils

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from layers import HalfNLHconv, MLP
import math 

class vhgae_encoder(nn.Module):
    def __init__(self, args, norm=None):
        super(vhgae_encoder, self).__init__()
        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = args.deepset_input_norm
        self.GPR = args.GPR
        self.LearnMask = args.LearnMask
        self.args = args
#         Now define V2EConvs[i], V2EConvs[i] for ith layers
#         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
#         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()
        if self.LearnMask:
            self.Importance = Parameter(torch.ones(norm.size()))

        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=args.num_features,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_classes,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=args.num_features,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            for _ in range(self.All_num_layers-1):
                self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
                self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))

        hidden = args.MLP_hidden
        self.encoder_mean_node = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.encoder_std_node = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.Softplus())
        self.encoder_mean_he = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.encoder_std_he = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.Softplus())

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self, data):
        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance*norm
        cidx = edge_index[1].min()
        edge_index[1] = edge_index[1] - cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)
        # x = F.dropout(x, p=0.2, training=self.training) # Input dropout
        for i, _ in enumerate(self.V2EConvs):
            # print(edge_index.dtype, edge_index[0].min(), edge_index[0].max(), edge_index[1].min(), edge_index[1].max())
            x_he = F.relu(self.V2EConvs[i](x, edge_index, norm, aggr=self.aggr))
#                 x = self.bnV2Es[i](x)
            x_x = F.dropout(x_he, p=self.dropout, training=self.training)
            x_node = F.relu(self.E2VConvs[i](x_x, reversed_edge_index, norm, self.aggr))
# #                 x = self.bnE2Vs[i](x)
            x = F.dropout(x_node, p=self.dropout, training=self.training)

        # x_f = F.dropout(x_node, p=self.dropout, training=self.training)

        x_mean_node = self.encoder_mean_node(x_node)
        x_std_node = self.encoder_std_node(x_node)
        gaussian_noise = torch.randn(x_mean_node.shape).to(x.device)
        x_node_final = gaussian_noise * x_std_node + x_mean_node

        x_mean_he = self.encoder_mean_he(x_he)
        x_std_he = self.encoder_std_he(x_he)
        gaussian_noise = torch.randn(x_mean_he.shape).to(x.device)
        x_he_final = gaussian_noise * x_std_he + x_mean_he
        return x_node_final, x_mean_node, x_std_node, x_he_final, x_mean_he, x_std_he


class vhgae_decoder(torch.nn.Module):
    def __init__(self, args):
        super(vhgae_decoder, self).__init__()
        hidden = args.MLP_hidden
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='mean')
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, x_node, x_mean_node, x_std_node, x_he, x_mean_he, x_std_he, num_ori_edge, edge_index, edge_index_neg, reward=None):
        # edge_pos_pred = self.sigmoid(self.decoder( x_node[edge_index[0]] * x_he[edge_index[1]] ))
        # edge_neg_pred = self.sigmoid(self.decoder( x_node[edge_index_neg[0]] * x_he[edge_index_neg[1]] ))
        i_1 = edge_index[0]
        i_2 = edge_index[1]
        x_2 = x_he[i_2]
        x_1 = x_node[i_1]
        
        edge_pos_pred = self.decoder( x_node[edge_index[0]] * x_he[edge_index[1]] )
        # edge_pos_pred = self.decoder( x_1 * x_2 )
        edge_neg_pred = self.decoder( x_node[edge_index_neg[0]] * x_he[edge_index_neg[1]] )

        '''
        # for link prediction
        import numpy as np
        from sklearn.metrics import roc_auc_score, average_precision_score
        edge_pred = torch.cat((edge_pos_pred, edge_neg_pred)).detach().cpu().numpy()
        edge_auroc = roc_auc_score(np.concatenate((np.ones(edge_pos_pred.shape[0]), np.zeros(edge_neg_pred.shape[0]))), edge_pred)
        edge_auprc = average_precision_score(np.concatenate((np.ones(edge_pos_pred.shape[0]), np.zeros(edge_neg_pred.shape[0]))), edge_pred)
        if True:
            return edge_auroc, edge_auprc
        # end link prediction
        '''
        # loss_edge_pos = self.bceloss( edge_pos_pred, torch.ones(edge_pos_pred.shape).to(edge_pos_pred.device) )
        # loss_edge_neg = self.bceloss( edge_neg_pred, torch.zeros(edge_neg_pred.shape).to(edge_neg_pred.device) )
        loss_edge_pos = self.celoss( edge_pos_pred, torch.ones(edge_pos_pred.shape[0], dtype=torch.long).to(edge_pos_pred.device) )
        loss_edge_neg = self.celoss( edge_neg_pred, torch.zeros(edge_neg_pred.shape[0], dtype=torch.long).to(edge_neg_pred.device) )
        loss_rec = loss_edge_pos + loss_edge_neg
        if not reward is None:
            loss_rec = loss_rec * reward

        # reference: https://github.com/DaehanKim/vgae_pytorch
        kl_divergence_node = - 0.5 * (1 + 2 * torch.log(x_std_node) - x_mean_node**2 - x_std_node**2).sum(dim=1).mean()
        kl_divergence_node = kl_divergence_node / x_node.size(0)
        kl_divergence_he = - 0.5 * (1 + 2 * torch.log(x_std_he) - x_mean_he**2 - x_std_he**2).sum(dim=1).mean()
        kl_divergence_he = kl_divergence_he / x_he.size(0)

        loss = (loss_rec + kl_divergence_node + kl_divergence_he).mean()
        return loss


class vhgae(torch.nn.Module):
    def __init__(self, encoder, decoder, args):
        super(vhgae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hard= True if args.hard>0 else False
        self.deg = args.deg

    def forward(self, data, reward=None):
        x_node, x_mean_node, x_std_node, x_he, x_mean_he, x_std_he = self.encoder(data)
        loss = self.decoder(x_node, x_mean_node, x_std_node, x_he, x_mean_he, x_std_he, data.num_ori_edge, data.edge_index, data.edge_index_neg, reward)
        return loss

    def generate(self, data, reward=None):
        edge_index = data.edge_index
        num_ori_edge = data.num_ori_edge
        src, dst = edge_index[0][:num_ori_edge], edge_index[1][:num_ori_edge]
        x_node, x_mean_node, x_std_node, x_he, x_mean_he, x_std_he = self.encoder(data)
        
        x_src, x_dst = x_node[src], x_he[dst]
        
        # prob = torch.einsum('nd,md->nmd', x, x)
        prob = x_src*x_dst
        prob = self.decoder.decoder(prob).squeeze()
        aug_edge_weight = F.gumbel_softmax(prob, tau=1, hard=self.hard)
        aug_keep_prob = aug_edge_weight[:,1]
        deg = 1-torch.mean(aug_keep_prob)
        if self.deg:
            loss = 0
        else:
            loss = self.decoder(x_node, x_mean_node, x_std_node, x_he, x_mean_he, x_std_he, data.num_ori_edge, data.edge_index, data.edge_index_neg, reward)
        self_prob = torch.ones(edge_index.shape[1]-num_ori_edge, dtype=torch.float).to(edge_index.device)
        aug_keep_prob = torch.cat([aug_keep_prob, self_prob])
        # aug_keep_prob = aug_keep_prob
        # aug_keep_prob = torch.ones(edge_index.shape[1], dtype=torch.float).to(edge_index.device)
        return loss, data, aug_keep_prob, deg

    def generate_only(self, data, reward=None):
        edge_index = data.edge_index
        num_ori_edge = data.num_ori_edge
        src, dst = edge_index[0][:num_ori_edge], edge_index[1][:num_ori_edge]
        x_node, x_mean_node, x_std_node, x_he, x_mean_he, x_std_he = self.encoder(data)
        x_src, x_dst = x_node[src], x_he[dst]
        # prob = torch.einsum('nd,md->nmd', x, x)
        prob = x_src*x_dst
        prob = self.decoder.decoder(prob).squeeze()
        aug_edge_weight = F.gumbel_softmax(prob, tau=1, hard=self.hard)
        aug_keep_prob = aug_edge_weight[:,1]
        deg = 1-torch.mean(aug_keep_prob)
        self_prob = torch.ones(edge_index.shape[1]-num_ori_edge, dtype=torch.float).to(edge_index.device)
        aug_keep_prob = torch.cat([aug_keep_prob, self_prob])
        # aug_keep_prob = aug_keep_prob
        # aug_keep_prob = torch.ones(edge_index.shape[1], dtype=torch.float).to(edge_index.device)
        return 0, data, aug_keep_prob, deg

class vhae(torch.nn.Module):
    def __init__(self, encoder, decoder, args):
        super(vhae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hard= True if args.hard>0 else False
        self.deg = args.deg

    def forward(self, data, reward=None):
        x_node, x_he = self.encoder(data)
        return 0

    def generate(self, data, reward=None):
        edge_index = data.edge_index
        num_ori_edge = data.num_ori_edge
        src, dst = edge_index[0][:num_ori_edge], edge_index[1][:num_ori_edge]
        x_node, x_he = self.encoder(data)
        
        x_src, x_dst = x_node[src], x_he[dst]
        
        # prob = torch.einsum('nd,md->nmd', x, x)
        prob = x_src*x_dst
        prob = self.decoder.decoder(prob).squeeze()
        aug_edge_weight = F.gumbel_softmax(prob, tau=1, hard=self.hard)
        aug_keep_prob = aug_edge_weight[:,1]
        deg = 1-torch.mean(aug_keep_prob)
        loss = 0
        self_prob = torch.ones(edge_index.shape[1]-num_ori_edge, dtype=torch.float).to(edge_index.device)
        aug_keep_prob = torch.cat([aug_keep_prob, self_prob])
        # aug_keep_prob = aug_keep_prob
        # aug_keep_prob = torch.ones(edge_index.shape[1], dtype=torch.float).to(edge_index.device)
        return loss, data, aug_keep_prob, deg

    def generate_only(self, data, reward=None):
        edge_index = data.edge_index
        num_ori_edge = data.num_ori_edge
        src, dst = edge_index[0][:num_ori_edge], edge_index[1][:num_ori_edge]
        x_node, x_he = self.encoder(data)
        x_src, x_dst = x_node[src], x_he[dst]
        # prob = torch.einsum('nd,md->nmd', x, x)
        prob = x_src*x_dst
        prob = self.decoder.decoder(prob).squeeze()
        aug_edge_weight = F.gumbel_softmax(prob, tau=1, hard=self.hard)
        aug_keep_prob = aug_edge_weight[:,1]
        deg = 1-torch.mean(aug_keep_prob)
        self_prob = torch.ones(edge_index.shape[1]-num_ori_edge, dtype=torch.float).to(edge_index.device)
        aug_keep_prob = torch.cat([aug_keep_prob, self_prob])
        # aug_keep_prob = aug_keep_prob
        # aug_keep_prob = torch.ones(edge_index.shape[1], dtype=torch.float).to(edge_index.device)
        return 0, data, aug_keep_prob, deg


class vhae_decoder(torch.nn.Module):
    def __init__(self, args):
        super(vhae_decoder, self).__init__()
        hidden = args.MLP_hidden
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='mean')
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, x_node, x_he):
        # edge_pos_pred = self.sigmoid(self.decoder( x_node[edge_index[0]] * x_he[edge_index[1]] ))
        # edge_neg_pred = self.sigmoid(self.decoder( x_node[edge_index_neg[0]] * x_he[edge_index_neg[1]] ))
        
        return 0
        
class vhae_encoder(nn.Module):
    def __init__(self, args, norm=None):
        super(vhae_encoder, self).__init__()
        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = args.deepset_input_norm
        self.GPR = args.GPR
        self.LearnMask = args.LearnMask
        self.args = args
#         Now define V2EConvs[i], V2EConvs[i] for ith layers
#         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
#         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()
        if self.LearnMask:
            self.Importance = Parameter(torch.ones(norm.size()))

        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=args.num_features,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_classes,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=args.num_features,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            for _ in range(self.All_num_layers-1):
                self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
                self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))

        hidden = args.MLP_hidden

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self, data):
        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance*norm
        cidx = edge_index[1].min()
        edge_index[1] = edge_index[1] - cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)
        x = F.dropout(x, p=0.2, training=self.training) # Input dropout
        for i, _ in enumerate(self.V2EConvs):
            # print(edge_index.dtype, edge_index[0].min(), edge_index[0].max(), edge_index[1].min(), edge_index[1].max())
            x_he = F.relu(self.V2EConvs[i](x, edge_index, norm, aggr=self.aggr))
#                 x = self.bnV2Es[i](x)
            x_x = F.relu(x_he)
            x_x = F.dropout(x_x, p=self.dropout, training=self.training)
            x_node = F.relu(self.E2VConvs[i](x_x, reversed_edge_index, norm, self.aggr))
            x = F.relu(x_node)
# #                 x = self.bnE2Vs[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # x_f = F.dropout(x_node, p=self.dropout, training=self.training)

        return x_node, x_he