import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os
from model.A2DCDR import A2DCDR
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)
sys.path.append('../utils')
from utils import torch_utils
import math
from model.adv_layer import ReverseLayerF

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class CrossTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        if self.opt["model"] == "A2DCDR":
            self.model = A2DCDR(opt)
            self.regs = eval(opt['regs'])
            self.decay = self.regs[0]
        else :
            print("please input right model name!")
            exit(0)

        self.criterion = nn.BCEWithLogitsLoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        self.epoch_rec_loss = []
        if self.opt["model"] == "A2DCDR":
            self.n_head = opt["fuse_heads"]
            self.tanh = nn.Tanh()

            if opt['cuda']:
                #target attention
                self.source_Q_nn = nn.Linear(opt["feature_dim"], opt["feature_dim"], bias=False).cuda()
                self.source_K_nn = nn.Linear(opt["feature_dim"], opt["feature_dim"], bias=False).cuda()
                self.source_V_nn = nn.Linear(opt["feature_dim"], opt["feature_dim"], bias=False).cuda()
                self.target_Q_nn = nn.Linear(opt["feature_dim"], opt["feature_dim"], bias=False).cuda()
                self.target_K_nn = nn.Linear(opt["feature_dim"], opt["feature_dim"], bias=False).cuda()
                self.target_V_nn = nn.Linear(opt["feature_dim"], opt["feature_dim"], bias=False).cuda()
                self.source_fc = nn.Linear(opt["feature_dim"], 1, bias=True).cuda()
                self.source_b = nn.Parameter(torch.rand(opt["feature_dim"])).cuda()
                self.target_fc = nn.Linear(opt["feature_dim"], 1, bias=True).cuda()
                self.target_b = nn.Parameter(torch.rand(opt["feature_dim"])).cuda()

            else:
                #target attention
                self.source_Q_nn = nn.Linear(opt["feature_dim"], opt["feature_dim"], bias=False)
                self.source_K_nn = nn.Linear(opt["feature_dim"], opt["feature_dim"], bias=False)
                self.source_V_nn = nn.Linear(opt["feature_dim"], opt["feature_dim"], bias=False)
                self.target_Q_nn = nn.Linear(opt["feature_dim"], opt["feature_dim"], bias=False)
                self.target_K_nn = nn.Linear(opt["feature_dim"], opt["feature_dim"], bias=False)
                self.target_V_nn = nn.Linear(opt["feature_dim"], opt["feature_dim"], bias=False)

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
        else:
            inputs = [Variable(b) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
        return user_index, item_index

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user = inputs[0]
            source_pos_item = inputs[1]
            source_neg_item = inputs[2]
            target_pos_item = inputs[3]
            target_neg_item = inputs[4]
        else:
            inputs = [Variable(b) for b in batch]
            user = inputs[0]
            source_pos_item = inputs[1]
            source_neg_item = inputs[2]
            target_pos_item = inputs[3]
            target_neg_item = inputs[4]
        return user, source_pos_item, source_neg_item, target_pos_item, target_neg_item

    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()
    
    def create_bpr_loss(self, users, pos_items, neg_items, users_pre, pos_items_pre, neg_items_pre):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        regularizer = (torch.norm(users_pre) ** 2 + torch.norm(pos_items_pre) ** 2 + 
                       torch.norm(neg_items_pre) ** 2) / 2
        regularizer = regularizer / self.opt['batch_size']
        
        mf_loss = torch.mean(torch.nn.functional.softplus(neg_scores-pos_scores))
        emb_loss = self.decay * regularizer
        return mf_loss, emb_loss

    def source_predict(self, batch):
        user_index, item_index = self.unpack_batch_predict(batch)

        if self.opt["model"] == "A2DCDR":
        #    print('source_predict  A2DCDR!')
        #    print('user_feature shape',user_feature.shape)
        #    print('item_feature shape',item_feature.shape)
           source_user_feature_share = self.my_index_select(self.source_user_share, user_index)
           source_user_feature_specific = self.my_index_select(self.source_user_specific, user_index)
           target_user_feature_share = self.my_index_select(self.target_user_share, user_index)
           item_feature = self.my_index_select(self.source_item, item_index)

        #    user_feature = source_user_feature_share + source_user_feature_specific + target_user_feature_share
        #    user_feature = user_feature.view(user_feature.size()[0], 1, -1)
        #    user_feature = user_feature.repeat(1, item_feature.size()[1], 1)
           
           user_features = torch.stack((source_user_feature_share, source_user_feature_specific, target_user_feature_share), dim=1)

        #    target_user_feature_specific = self.my_index_select(self.target_user_specific, user_index)
        #    user_features = torch.stack((source_user_feature_share, source_user_feature_specific, target_user_feature_share, target_user_feature_specific), dim=1)
           user_feature = self.target_attention(item_feature, user_features, domain='source')
        else:
            user_feature = self.my_index_select(self.source_user, user_index)
            item_feature = self.my_index_select(self.source_item, item_index)
            user_feature = user_feature.view(user_feature.size()[0], 1, -1)
            user_feature = user_feature.repeat(1, item_feature.size()[1], 1)

        # user_feature = user_feature.view(user_feature.size()[0], 1, -1)
        # user_feature = user_feature.repeat(1, item_feature.size()[1], 1)
        score = self.model.source_predict_dot(user_feature, item_feature)
        return score.view(score.size()[0], score.size()[1])

    def target_predict(self, batch):
        user_index, item_index = self.unpack_batch_predict(batch)

        if self.opt["model"] == "A2DCDR":
            # print('target_predict  A2DCDR!')

            source_user_feature_share = self.my_index_select(self.source_user_share, user_index)
            target_user_feature_share = self.my_index_select(self.target_user_share, user_index)
            target_user_feature_specific = self.my_index_select(self.target_user_specific, user_index)
            item_feature = self.my_index_select(self.target_item, item_index)

            user_features = torch.stack((target_user_feature_share, target_user_feature_specific, source_user_feature_share), dim=1)

            user_feature = self.target_attention(item_feature, user_features, domain='target')
        else:
            user_feature = self.my_index_select(self.target_user, user_index)
            item_feature = self.my_index_select(self.target_item, item_index)
            user_feature = user_feature.view(user_feature.size()[0], 1, -1)
            user_feature = user_feature.repeat(1, item_feature.size()[1], 1)

        
        score = self.model.target_predict_dot(user_feature, item_feature)
        return score.view(score.size()[0], score.size()[1])

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def evaluate_embedding(self, source_UV, source_VU, target_UV, target_VU, source_adj=None, target_adj=None):
        if self.opt["model"] == "A2DCDR":
            # self.source_user, self.source_item, self.target_user, self.target_item = self.model.wramup(source_UV, source_VU, target_UV, target_VU)
            self.source_user_share, self.source_user_specific, self.source_item, self.target_user_share, self.target_user_specific, self.target_item = self.model(source_UV,source_VU, target_UV,target_VU)
            return self.source_user_share, self.source_user_specific, self.target_user_share, self.target_user_specific
        else:
            self.source_user, self.source_item, self.target_user, self.target_item = self.model(source_UV, source_VU, target_UV, target_VU)

    def for_bcelogit(self, x):
        y = 1 - x
        return torch.cat((x,y), dim = -1)
    
    def split(self, tensor):
        #https://github.com/hyunwoongko/transformer/blob/master/models/layers/multi_head_attention.py
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor    
    
    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
    
    def target_attention(self, tar_features, input2, domain='source'):
        batch_size, length, d_model = input2.size()
        # Linear projections
        if len(tar_features.size()) == 3:
            Q_features = tar_features  #[batch_size, 1000, d_model]
        else:
            Q_features = torch.unsqueeze(tar_features, 1) #[batch_size, 1, d_model]
        if domain == 'source':
            # Q, K, V = self.source_Q_nn(Q_features), self.source_K_nn(input2), input2
            Q, K, V = Q_features, input2, input2
        elif domain == 'target':
            # Q, K, V = self.target_Q_nn(Q_features), self.target_K_nn(input2), input2
            Q, K, V = Q_features, input2, input2
        # 2. split tensor by number of heads
        ## [batch_size, head, length, d_tensor]
        # Q, K, V = self.split(Q), self.split(K), self.split(V)
        
        #attention
        # broadcasting
        # if domain == 'source':
        #     score = self.source_fc(self.tanh(torch.unsqueeze(Q, 2) + torch.unsqueeze(K, 1) +self.source_b)).squeeze(dim=-1)
        # elif domain == 'target':
        #     score = self.target_fc(self.tanh(torch.unsqueeze(Q, 2) + torch.unsqueeze(K, 1) +self.target_b)).squeeze(dim=-1)
        #attention
        K_t = K.transpose(1, 2)  # transpose  
        score = (Q @ K_t) / math.sqrt(d_model)  # scaled dot product  # [batch_size, head, 1, length]
        score = nn.Softmax(dim=-1)(score)
        out = score @ V   #[batch_size, head, 1, d_tensor]
        # 4. concat and pass to linear layer
        # out = self.concat(out) #[batch_size, 1, d_model]
        if len(tar_features.size()) == 2:
            out = torch.squeeze(out, 1) #[batch_size, d_model]
        return out

    def reconstruct_graph(self, batch, source_UV, source_VU, target_UV, target_VU, source_adj=None, target_adj=None, epoch = 100):
        self.model.train()
        self.optimizer.zero_grad()

        user, source_pos_item, source_neg_item, target_pos_item, target_neg_item = self.unpack_batch(batch)


        if self.opt["model"] == "A2DCDR":
            if epoch<10:
                self.source_user, self.source_item, self.target_user, self.target_item = self.model.wramup(source_UV,source_VU,target_UV,target_VU)
                source_user_feature = self.my_index_select(self.source_user, user)
                source_item_pos_feature = self.my_index_select(self.source_item, source_pos_item)
                source_item_neg_feature = self.my_index_select(self.source_item, source_neg_item)

                target_user_feature = self.my_index_select(self.target_user, user)
                target_item_pos_feature = self.my_index_select(self.target_item, target_pos_item)
                target_item_neg_feature = self.my_index_select(self.target_item, target_neg_item)

                pos_source_score = self.model.source_predict_dot(source_user_feature, source_item_pos_feature)
                neg_source_score = self.model.source_predict_dot(source_user_feature, source_item_neg_feature)
                pos_target_score = self.model.target_predict_dot(target_user_feature, target_item_pos_feature)
                neg_target_score = self.model.target_predict_dot(target_user_feature, target_item_neg_feature)
            else:
                self.source_user_share, self.source_user_specific, self.source_item, self.target_user_share, self.target_user_specific, self.target_item = self.model(source_UV,source_VU, target_UV,target_VU)

                source_user_feature_share = self.my_index_select(self.source_user_share, user)
                source_user_feature_specific = self.my_index_select(self.source_user_specific, user)
                source_item_pos_feature = self.my_index_select(self.source_item, source_pos_item)
                source_item_neg_feature = self.my_index_select(self.source_item, source_neg_item)

                target_user_feature_share = self.my_index_select(self.target_user_share, user)
                target_user_feature_specific = self.my_index_select(self.target_user_specific, user)
                target_item_pos_feature = self.my_index_select(self.target_item, target_pos_item)
                target_item_neg_feature = self.my_index_select(self.target_item, target_neg_item)

                target_user_features = torch.stack((target_user_feature_share, target_user_feature_specific, source_user_feature_share), dim=1)
                target_user_pos_feature = self.target_attention(target_item_pos_feature, target_user_features, domain='target')
                target_user_neg_feature = self.target_attention(target_item_neg_feature, target_user_features, domain='target')

                source_user_features = torch.stack((source_user_feature_share, source_user_feature_specific, target_user_feature_share), dim=1)
                source_user_pos_feature = self.target_attention(source_item_pos_feature, source_user_features, domain='source')
                source_user_neg_feature = self.target_attention(source_item_neg_feature, source_user_features, domain='source')


                pos_source_score = self.model.source_predict_dot(source_user_pos_feature, source_item_pos_feature)
                neg_source_score = self.model.source_predict_dot(source_user_neg_feature, source_item_neg_feature)
                pos_target_score = self.model.target_predict_dot(target_user_pos_feature, target_item_pos_feature)
                neg_target_score = self.model.target_predict_dot(target_user_neg_feature, target_item_neg_feature)
        else:
            if epoch<10:
                self.source_user, self.source_item, self.target_user, self.target_item = self.model.wramup(source_UV,source_VU,target_UV,target_VU)
            else:
                self.source_user, self.source_item, self.target_user, self.target_item = self.model(source_UV,source_VU, target_UV,target_VU)

            source_user_feature = self.my_index_select(self.source_user, user)
            source_item_pos_feature = self.my_index_select(self.source_item, source_pos_item)
            source_item_neg_feature = self.my_index_select(self.source_item, source_neg_item)

            target_user_feature = self.my_index_select(self.target_user, user)
            target_item_pos_feature = self.my_index_select(self.target_item, target_pos_item)
            target_item_neg_feature = self.my_index_select(self.target_item, target_neg_item)


            pos_source_score = self.model.source_predict_dot(source_user_feature, source_item_pos_feature)
            neg_source_score = self.model.source_predict_dot(source_user_feature, source_item_neg_feature)
            pos_target_score = self.model.target_predict_dot(target_user_feature, target_item_pos_feature)
            neg_target_score = self.model.target_predict_dot(target_user_feature, target_item_neg_feature)

        pos_labels, neg_labels = torch.ones(pos_source_score.size()), torch.zeros(
            pos_source_score.size())

        if self.opt["cuda"]:
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        if self.opt["model"] == "A2DCDR":
            loss = [
               self.criterion(pos_source_score, pos_labels) + \
               self.criterion(neg_source_score, neg_labels) + \
               self.criterion(pos_target_score, pos_labels) + \
               self.criterion(neg_target_score, neg_labels) + \
               self.model.mmd_loss + \
               self.model.reverse_mmd_loss + \
               self.model.loss_recon + \
               self.model.loss_reg , self.model.mmd_loss, self.model.reverse_mmd_loss, self.model.loss_reg ,self.model.loss_recon]
            
            loss[0].backward()
            self.optimizer.step()
            return tuple([x.item() for x in loss])
        else:
            loss = self.criterion(pos_source_score, pos_labels) + \
               self.criterion(neg_source_score, neg_labels) + \
               self.criterion(pos_target_score, pos_labels) + \
               self.criterion(neg_target_score, neg_labels) + \
               self.model.source_specific_GNN.encoder[-1].kld_loss + \
               self.model.target_specific_GNN.encoder[-1].kld_loss + self.model.kld_loss

            loss.backward()
            self.optimizer.step()
            return loss.item()
