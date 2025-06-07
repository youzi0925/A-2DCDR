import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.GCN import GCN
from model.GCN import VGAE
from torch.autograd import Variable
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class singleLGCN(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt, domain):
        super(singleLGCN, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.dropout = opt["dropout"]
        self.emb_dim = opt["feature_dim"]
        if domain == 'source':
           self.num_users, self.num_items = opt["source_user_num"], opt["source_item_num"]
        elif domain == 'target':
           self.num_users, self.num_items = opt["target_user_num"], opt["target_item_num"]

        self.user_union1 = nn.Linear(int(opt["feature_dim"]), int(opt["feature_dim"]/2))
        self.user_union2 = nn.Linear(int(opt["feature_dim"]/2), int(opt["feature_dim"]))
        self.union = nn.Linear(opt["feature_dim"], int(opt["feature_dim"]/2))

        self.encoder = []
        for i in range(self.layer_number):
            self.encoder.append(DLGCNLayer(opt))
        self.encoder = nn.ModuleList(self.encoder)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x._indices().t()
        values = x._values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, adj, keep_prob):
        graph = self.__dropout_x(adj, keep_prob)
        return graph
    
    def forward(self, ufea, vfea, UV_adj, VU_adj):
        """
        propagate methods for lightGCN
        """       
        learn_user = ufea
        learn_item = vfea
        for layer in self.encoder:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_item = F.dropout(learn_item, self.dropout, training=self.training)
            learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj)
        
        return learn_user, learn_item
    
    def forward_user(self, ufea, UV_adj, VU_adj):
        # User_ho = self.aggregation(ufea, VU_adj)
        # User_ho = self.aggregation(User_ho, UV_adj)
        # User = torch.cat((User_ho, ufea), dim=1)
        User = F.relu(self.user_union1(ufea))
        User = self.user_union2(User)
        return F.relu(User)
        # return User
    
class DLGCNLayer(nn.Module):
    def __init__(self, opt):
        super(DLGCNLayer, self).__init__()
        self.opt = opt
        self.dropout = opt["dropout"]
        self.aggregation = self.neighbor_aggregation
        self.user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x._indices().t()
        values = x._values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, adj, keep_prob):
        graph = self.__dropout_x(adj, keep_prob)
        return graph

    def neighbor_aggregation(self, features, adj):
        # if self.dropout:
        #     if self.training:
        #         # print("droping")
        #         g_droped = self.__dropout(adj, 1-self.dropout)
        #     else:
        #         g_droped = adj       
        # else:
        #     g_droped = adj
        # return torch.spmm(g_droped, features)
        return torch.spmm(adj, features)



    def forward(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.aggregation(ufea, VU_adj)
        Item_ho = self.aggregation(vfea, UV_adj)
        User_ho = self.aggregation(User_ho, UV_adj)
        Item_ho = self.aggregation(Item_ho, VU_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        Item = torch.cat((Item_ho, vfea), dim=1)
        User = self.user_union(User)
        Item = self.item_union(Item)
        return F.relu(User), F.relu(Item)
        # return User, Item
        # return User_ho, Item_ho   


