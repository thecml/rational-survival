import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import warnings

# Rational extration
from rational.generator import Generator

class Encoder(nn.Module):
    def __init__(self, in_features: int):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        """
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.in_features = in_features
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(self.in_features, 1)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not mask is None:
            x_mask = x * mask.unsqueeze(-1)
        x_mask = x_mask.squeeze(-1)
        # logits = self.dropout(self.relu(self.fc(x_mask)))
        logits = self.fc(x_mask)
        
        return logits

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features})")

    def get_name(self):
        return self._get_name()
    
class MHLR_rational_Model(nn.Module):
    def __init__(self, input_size, num_time_bins, GeneratorArgs):
        super(MHLR_rational_Model, self).__init__()
        self.num_time_bins = num_time_bins
        self.generators = [Generator(num_features, args=GeneratorArgs) for num in range(num_time_bins)]
        self.encoder_list = [Encoder(input_size) for num in range(num_time_bins)]
        self.mlps = nn.ModuleList(self.encoder_list)
        self.masks = []
        self.GeneratorArgs = GeneratorArgs

    def forward(self, x):
        self.masks = []
        if self.GeneratorArgs.get_rationales:
            for gen in self.generators:
                mask, _ = gen(x)
                self.masks.append(mask)
        else:
            for gen in self.generators:
                self.masks.append(None)
        
        x = [self.encoder_list[i](x, self.masks[i]) for i in range(len(self.encoder_list))]
        outputs = torch.cat(x, axis=1)
        return outputs