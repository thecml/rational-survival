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
            x_mask = x.squeeze(-1)
        else:
            x_mask = x.squeeze(-1)
        # logits = self.dropout(self.relu(self.fc(x_mask)))
        logits = self.fc(x_mask)
        logits = self.relu(logits)
        logits = self.dropout(logits)
        
        return logits

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features})")

    def get_name(self):
        return self._get_name()
    
class MHLR_rational_Model(nn.Module):
    def __init__(self, num_features, num_time_bins, GeneratorArgs):
        super(MHLR_rational_Model, self).__init__()
        self.num_time_bins = num_time_bins
        self.generators = nn.ModuleList([Generator(num_features, args=GeneratorArgs) for num in range(num_time_bins)])
        self.encoder_list = [Encoder(num_features) for num in range(num_time_bins)]
        self.mlps = nn.ModuleList(self.encoder_list)
        self.masks = []
        self.GeneratorArgs = GeneratorArgs

    def forward(self, x, test=False):
        self.masks = []
        if self.GeneratorArgs.get_rationales:
            for gen in self.generators:
                mask, _ = gen(x, test)
                self.masks.append(mask)
        else:
            for gen in self.generators:
                self.masks.append(None)
        
        x = [self.encoder_list[i](x, self.masks[i]) for i in range(len(self.encoder_list))]
        outputs = torch.cat(x, axis=1)
        return outputs
        
class MHLR(nn.Module):
    """Multi-task logistic regression for individualised
    survival prediction.

    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time
    interval.

    Note that a slightly more efficient reformulation is used here, first
    proposed in [2]_.

    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival
    distributions as a sequence of dependent regressors’, in Advances in neural
    information processing systems 24, 2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn
    Consumer-Specific Reservation Price Distributions’, Master's thesis,
    University of Alberta, Edmonton, AB, 2015.
    """

    def __init__(self, in_features: int, num_time_bins: int):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        """
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1 # + extra time bin [max_time, inf)

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features,
                                                     self.num_time_bins - 1))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        # self.register_buffer(
        #     "G",
        #     torch.tril(
        #         torch.ones(self.num_time_bins - 1,
        #                    self.num_time_bins,
        #                    requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins)
            The predicted time logits.
        """
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        # return torch.matmul(out, self.G)
        # out = F.pad(out, (0, 1), 'constant', 0)
        return out

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_time_bins})")
    
    def get_name(self):
        return self._get_name()