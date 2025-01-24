import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from params import args
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"


#pretrain_model
class SSL_LTVModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(New_LTVModel, self).__init__()
        self.project_com = nn.Linear(input_dim, 256)
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.augment_net = Augment_net(256,64,2)
        self.embedding_net1 = nn.Linear(256, 128)
        self.embedding_net2 = nn.Linear(128, 64)
        self.p_layer = nn.Linear(64, 1)
        self.mu_layer = nn.Linear(64, 1)
        self.sigma_layer = nn.Linear(64, 1)
    def forward(self,com_feat):
        batch_feature = self.project_com(com_feat)
        aug_feature = self.augment_net(batch_feature)
        cl_loss = infor_nce_loss(batch_feature,aug_feature,args.temp)
        # cl_loss = cal_infonce_loss(batch_feature, aug_feature,com_feat_proj, args.temp)
        # confuse_feature = batch_feature
        batch_feature = self.relu(self.embedding_net1(batch_feature))
        batch_feature = self.relu(self.embedding_net2(batch_feature))
        p = self.p_layer(batch_feature)
        mu = self.mu_layer(batch_feature)
        sigma = self.sigma_layer(batch_feature)
        out = torch.concat([p, mu, sigma], axis=-1)
        return out, cl_loss

    def get_test(self,com_feat):
        batch_feature = self.project_com(com_feat)

        batch_feature = self.relu(self.embedding_net1(batch_feature))
        batch_feature = self.relu(self.embedding_net2(batch_feature))
        p = self.p_layer(batch_feature)
        mu = self.mu_layer(batch_feature)
        sigma = self.sigma_layer(batch_feature)
        out = torch.concat([p, mu, sigma], axis=-1)
        return out
    def get_scores(self,com_feat):
        batch_feature = self.project_com(com_feat)
        batch_feature = self.relu(self.embedding_net1(batch_feature))
        batch_feature = self.relu(self.embedding_net2(batch_feature))
        return batch_feature

#fine-tuned model

class SSL_Finetune_LTVModel(nn.Module):
    def __init__(self, input_dim,hidden_dim=64):
        super(SSL_Finetune_LTVModel, self).__init__()
        self.project_com = nn.Linear(input_dim, 256)
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.embedding_net1 = nn.Linear(256, 128)
        self.embedding_net2 = nn.Linear(128, 64)
        self.p_layer1 = nn.Linear(64, 1)
        self.mu_layer1 = nn.Linear(64, 1)
        self.sigma_layer1 = nn.Linear(64, 1)

        self.p_layer2 = nn.Linear(64, 1)
        self.mu_layer2 = nn.Linear(64, 1)
        self.sigma_layer2 = nn.Linear(64, 1)

    def forward(self,com_feat,batch_size):
        com_feat_proj = self.project_com(com_feat)
        hidden_feature = self.relu(self.embedding_net1(com_feat_proj))
        hidden_feature = self.relu(self.embedding_net2(hidden_feature))
        s_x = hidden_feature[:batch_size]
        t_x = hidden_feature[batch_size:]
        p1 = self.p_layer1(s_x)
        mu1 = self.mu_layer1(s_x)
        sigma1 = self.sigma_layer1(s_x)
        p2 = self.p_layer2(t_x)
        mu2 = self.mu_layer2(t_x)
        sigma2 = self.sigma_layer2(t_x)
        out1 = torch.concat([p1, mu1, sigma1], axis=-1)
        out2 = torch.concat([p2, mu2, sigma2], axis=-1)
        return hidden_feature,out1,out2
        # return com_feat_proj, out1, out2,s_x,t_x
    def get_scores(self,com_feat):
        com_feat_proj = self.project_com(com_feat)
        hidden_feature = self.relu(self.embedding_net1(com_feat_proj))
        hidden_feature = self.relu(self.embedding_net2(hidden_feature))
        p2 = self.p_layer2(hidden_feature)
        mu2 = self.mu_layer2(hidden_feature)
        sigma2 = self.sigma_layer2(hidden_feature)
        out2 = torch.concat([p2, mu2, sigma2], axis=-1)
        return out2




class Augment_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Augment_net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return h

    def reparameterize(self, h):
        p = F.gumbel_softmax(h,hard=True)
        return p

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return h3

    def forward(self, x):
        h = self.encode(x)
        h1=self.fc2(h)
        p = self.reparameterize(h1)
        real_sample = p[:, 0]
        h = h*real_sample.view(-1,1)
        return self.decode(h)

def infor_nce_loss(emb1,emb2,temp=1.0):
    normed_embeds1 = emb1 / torch.sqrt(1e-8 + emb1.square().sum(-1, keepdim=True))
    normed_embeds2 = emb2 / torch.sqrt(1e-8 + emb2.square().sum(-1, keepdim=True))
    sim_matrix = torch.matmul(normed_embeds1,normed_embeds2.T)

    # 归一化相似度矩阵
    sim_matrix = sim_matrix / temp

    labels = torch.arange(len(emb1)).unsqueeze(0).expand(len(emb1), -1).eq(torch.arange(len(emb1)).unsqueeze(-1).expand(-1, len(emb1)))

    # 计算正样本的相似度
    # positive_samples = torch.diag(sim_matrix, diagonal=0)
    positive_logits = sim_matrix[labels].view(-1, 1)

    # 计算负样本的相似度
    negative_logits = sim_matrix[~labels].view(len(emb1), -1)
    # negative_samples = sim_matrix[torch.arange(len(emb1)), torch.arange(len(emb2))].view(-1, 1).expand(-1,len(emb1) - 1)



    # 计算损失
    logits = torch.cat([positive_logits, negative_logits], dim=1)
    labels = torch.zeros(len(emb1), dtype=torch.long).to(emb1.device)
    loss = F.cross_entropy(logits, labels)

    return loss
