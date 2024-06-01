import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

from rational import mlp
from rational import learn

class Generator(nn.Module):

    def __init__(self, n_features, args):
        super(Generator, self).__init__()
        self.args = args
        self.mlp = mlp.MLP(n_features, args)

        self.z_dim = 2

        self.hidden = nn.Linear(1, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)

    def z_forward(self, x):
        '''
        Returns prob of each token being selected
        '''
        logits = self.hidden(x) # TODO: Consider using the MLP here
        probs = learn.gumbel_softmax(logits, self.args.gumbel_temprature, self.args.cuda)
        z = probs[:,:,1] # shape = [B, N_FT]
        return z

    def forward(self, x_indx):
        '''
        Given input x_indx of dim (batch, length), return z (batch, length) such that z
        can act as element-wise mask on x
        '''
        x = x_indx
        if self.args.cuda:
            x = x.cuda()
        
        activ = x
        z = self.z_forward(F.relu(activ))
        mask = self.sample(z)
        return mask, z

    def sample(self, z):
        '''
        Get mask from probablites at each token. Use gumbel
        softmax at train time, hard mask at test time
        '''
        mask = z
        if self.training:
            mask = z
        else:
            mask = learn.get_hard_mask(z)
        return mask

    def loss(self, mask, x_indx):
        '''
        Compute the generator specific costs, i.e selection cost, continuity cost, and global vocab cost
        '''
        selection_cost = torch.mean(torch.sum(mask, dim=0))
        return selection_cost
