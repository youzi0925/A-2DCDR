import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.singleLGCN import singleLGCN
from model.adv_layer import ReverseLayerF
from model.grl import WarmStartGradientReverseLayer
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import math
from model.CLUB import CLUB


class A2DCDR(nn.Module):
    def __init__(self, opt):
        super(A2DCDR, self).__init__()
        self.opt=opt

        self.dropout = opt["dropout"]
        self.w_gamma_a = opt["w_gamma_a"]
        self.w_gamma_b = opt["w_gamma_b"]
        self.w_beta_a = opt["w_beta_a"]
        self.w_beta_b = opt["w_beta_b"]
        self.w_alpha = opt["w_alpha"]

        # self.user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])

        self.source_user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding = nn.Embedding(opt["target_user_num"], opt["feature_dim"])
        self.source_item_embedding = nn.Embedding(opt["source_item_num"], opt["feature_dim"])
        self.target_item_embedding = nn.Embedding(opt["target_item_num"], opt["feature_dim"])
        self.source_user_embedding_share = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding_share = nn.Embedding(opt["target_user_num"], opt["feature_dim"])

        self.source_specific_GNN = singleLGCN(opt, 'source')
        self.source_share_GNN = singleLGCN(opt, 'source')

        self.target_specific_GNN = singleLGCN(opt, 'target')
        self.target_share_GNN = singleLGCN(opt, 'target')
        self.target_GNN= singleLGCN(opt, 'target')
        self.club = CLUB(opt)
        self.loss_MSE = torch.nn.MSELoss()

        # reconstructor
        self.source_feature_reconstructor = nn.Sequential( # test reconstructor

            # nn.Linear(opt["feature_dim"]*2, opt["feature_dim"]*2),
            # nn.ReLU(),

            # nn.Linear(1024, 512),
            # nn.ReLU(),

            nn.Linear(opt["feature_dim"]*2, opt["feature_dim"]*2),
            nn.ReLU()
        )
        self.target_feature_reconstructor = nn.Sequential( # test reconstructor

            # nn.Linear(opt["feature_dim"]*2, opt["feature_dim"]*2),
            # nn.ReLU(),

            # nn.Linear(1024, 512),
            # nn.ReLU(),

            nn.Linear(opt["feature_dim"]*2, opt["feature_dim"]*2),
            nn.ReLU()
        )


        self.user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.source_user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.target_user_index = torch.arange(0, self.opt["target_user_num"], 1)
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1)

        if self.opt["cuda"]:
            self.user_index = self.user_index.cuda()
            self.source_user_index = self.source_user_index.cuda()
            self.target_user_index = self.target_user_index.cuda()
            self.source_item_index = self.source_item_index.cuda()
            self.target_item_index = self.target_item_index.cuda()

    def source_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def target_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output
    
    def _mmd_linear(self, f_of_X, f_of_Y):
        #https://github.com/syorami/DDC-transfer-learning
        delta = f_of_X - f_of_Y
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss
        # loss = 0.0
        # delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        # loss = delta.dot(delta.T)
        # return loss
    
    def _orthogonal_loss(self, src, trg):
        return torch.mean(torch.sum(src*trg, dim=1)**2)

    def _l2_rec(self, src, trg):
        recon_nn1 = nn.Linear(src.shape[1], trg.shape[1], bias=True)
        recon_nn2 = nn.Linear(trg.shape[1], trg.shape[1], bias=True)
        if self.opt["cuda"]:
            recon_nn1 = recon_nn1.cuda()
            recon_nn2 = recon_nn2.cuda()
        src = recon_nn2(F.relu(recon_nn1(src)))
        return torch.mean((src - trg)**2)
                    
    def forward(self, source_UV, source_VU, target_UV, target_VU):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)
        source_user_share = self.source_user_embedding_share(self.source_user_index)
        target_user_share = self.target_user_embedding_share(self.target_user_index)

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item, source_UV, source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item, target_UV, target_VU)

        source_learn_share_user, _ = self.source_share_GNN(source_user_share, source_item, source_UV, source_VU)
        target_learn_share_user, _ = self.target_share_GNN(target_user_share, target_item, target_UV, target_VU)

        reverse_target_learn_specific_user = ReverseLayerF.apply(target_learn_specific_user, 1.0)
        # reverse_target_learn_specific_user = self.grl(target_learn_specific_user)
        reverse_target_learn_specific_user = self.target_GNN.forward_user(reverse_target_learn_specific_user, target_UV, target_VU)


        self.mmd_loss = self.w_alpha * self._mmd_linear(source_learn_share_user, target_learn_share_user) 
        self.reverse_mmd_loss = self.w_alpha * self._mmd_linear(source_learn_share_user, reverse_target_learn_specific_user)


        # self.loss_reg = self.w_beta_b * self._orthogonal_loss(target_learn_specific_user, target_learn_share_user) + self.w_beta_a * self._orthogonal_loss(source_learn_specific_user, source_learn_share_user)
        self.loss_reg = self.w_beta_b * self.club.learning_loss(target_learn_specific_user, target_learn_share_user) + self.w_beta_a * self.club.learning_loss(source_learn_specific_user, source_learn_share_user)

        # self.loss_recon = self.w_gamma_b * self._l2_rec(target_learn_share_user + target_learn_specific_user , target_user+target_user_share) +  self.w_gamma_a * self._l2_rec(source_learn_share_user + source_learn_specific_user , source_user+source_user_share)

        # source_rec = self.source_feature_reconstructor(torch.cat((source_learn_share_user,source_learn_specific_user), dim=-1))
        # target_rec = self.target_feature_reconstructor(torch.cat((target_learn_share_user,target_learn_specific_user), dim=-1))

        # self.loss_recon = 0.1*self.w_l2 * self.loss_MSE(torch.cat((source_user,source_user_share), dim=-1), source_rec) + 0.9*self.w_l2 * self.loss_MSE(torch.cat((target_user,target_user_share), dim=-1), target_rec)

        source_rec = self.source_feature_reconstructor(torch.cat((source_learn_share_user,source_learn_specific_user), dim=-1))
        target_rec = self.target_feature_reconstructor(torch.cat((target_learn_share_user,target_learn_specific_user), dim=-1))

        self.loss_recon = self.w_gamma_a * self.loss_MSE(torch.cat((source_user,source_user_share), dim=-1), source_rec) + self.w_gamma_b * self.loss_MSE(torch.cat((target_user,target_user_share), dim=-1), target_rec)

        # self.loss_recon = self.w_gamma_b * self._l2_rec(torch.cat((target_learn_share_user,target_learn_specific_user), dim=-1) , torch.cat((target_user,target_user_share), dim=-1)) +  self.w_gamma_a * self._l2_rec(torch.cat((source_learn_share_user,source_learn_specific_user), dim=-1) , torch.cat((source_user,source_user_share), dim=-1))

        # return source_final_user, source_learn_specific_item, target_final_user, target_learn_specific_item
        return source_learn_share_user, source_learn_specific_user, source_learn_specific_item, target_learn_share_user, target_learn_specific_user, target_learn_specific_item


    def wramup(self, source_UV, source_VU, target_UV, target_VU):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item,source_UV, source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item,target_UV, target_VU)

        self.mmd_loss = torch.tensor(0.0)
        self.loss_reg = torch.tensor(0.0)
        self.loss_recon = torch.tensor(0.0)
        self.reverse_mmd_loss = torch.tensor(0.0)

        return source_learn_specific_user, source_learn_specific_item, target_learn_specific_user, target_learn_specific_item
    
