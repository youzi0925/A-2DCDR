import torch
import torch.nn as nn

class CLUB(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, opt):
        super(CLUB, self).__init__()
        self.nfeat, self.nhid = opt["feature_dim"], opt["hidden_dim"] 
        self.p_mu = nn.Sequential(nn.Linear(self.nfeat, self.nhid),
                                       nn.ReLU(),
                                       nn.Linear(self.nhid, self.nfeat))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(self.nfeat, self.nhid),
                                       nn.ReLU(),
                                       nn.Linear(self.nhid, self.nfeat),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
