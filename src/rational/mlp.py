import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

class MLP(nn.Module):
    def __init__(self, n_features, args):
        super(MLP, self).__init__()

        self.args = args
        self.layers = nn.ModuleList()

        input_dim = n_features
        
        for layer in range(args.num_layers):
            dense_layers = nn.ModuleList()
            if layer == 0:
                in_features = input_dim  # Use the input dimension for the first layer
            else:
                in_features = args.hidden_dim  # Adjust input features based on previous layers
            
            # Out features remain same as number of filters
            out_features = n_features
            new_dense = nn.Linear(in_features, out_features)
            self.add_module('layer_' + str(layer) + '_dense_', new_dense)
            dense_layers.append(new_dense)
            
        self.layers.append(dense_layers)

    def forward(self, x):
        batch_size = x.size(0)
        
        for dense_layer in self.layers:
            next_activ = []
            for dense in dense_layer:
                out = F.relu(dense(x))
                next_activ.append(out)
            
            x = torch.cat(next_activ, dim=1)  # Concatenate outputs
        
        return x



	