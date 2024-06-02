import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

class Encoder(nn.Module):

    def __init__(self, n_features, args):
        super(Encoder, self).__init__()
        self.args = args
        self.fc = nn.Linear(n_features, args.num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x_indx, mask=None):
        x = x_indx
        
        if self.args.cuda:
            x = x.cuda()
        if not mask is None:
            x = x * mask.unsqueeze(-1)
        
        x = x.squeeze(-1)
        logits = self.dropout(self.relu(self.fc(x)))
    
        return logits
