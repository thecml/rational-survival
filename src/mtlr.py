import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from typing import List, Tuple, Optional, Union
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from utility import reformat_survival
from rational import learn
from collections import defaultdict

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def get_optimizer(models, args):
    params = []
    for model in models:
        params.extend([param for param in model.parameters() if param.requires_grad])
    return torch.optim.Adam(params, lr=args.init_lr,  weight_decay=args.weight_decay)

def make_optimizer(opt_cls, model, **kwargs):
    """Creates a PyTorch optimizer for MTLR training."""
    params_dict = dict(model.named_parameters())
    weights = [v for k, v in params_dict.items() if "mtlr" not in k and "bias" not in k]
    biases = [v for k, v in params_dict.items() if "bias" in k]
    mtlr_weights = [v for k, v in params_dict.items() if "mtlr_weight" in k]
    # Don't use weight decay on the biases and MTLR parameters, which have
    # their own separate L2 regularization
    optimizer = opt_cls([
        {"params": weights},
        {"params": biases, "weight_decay": 0.},
        {"params": mtlr_weights, "weight_decay": 0.},
    ], **kwargs)
    return optimizer

class mtlr(nn.Module):
    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
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
        self.config = config
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features,
                                                     self.num_time_bins - 1))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
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
    
class Encoder(nn.Module):
    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
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
        self.config = config
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(self.in_features, 2)

        #self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features,
        #                                             self.num_time_bins - 1))
        #self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        #self.register_buffer(
        #    "G",
        #    torch.tril(
        #        torch.ones(self.num_time_bins - 1,
        #                   self.num_time_bins,
        #                   requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not mask is None:
            x_mask = x * mask.unsqueeze(-1)
        x_mask = x_mask.squeeze(-1)
        logits = self.dropout(self.relu(self.fc(x_mask)))
        return logits
        
        #out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        #return out

    def reset_parameters(self):
        """Resets the model parameters."""
        #nn.init.xavier_normal_(self.mtlr_weight)
        #nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_time_bins})")

    def get_name(self):
        return self._get_name()
    

def mtlr_survival(
        logits: torch.Tensor,
        with_sample: bool = True
) -> torch.Tensor:
    # TODO: do not reallocate G in every call
    if with_sample:
        assert logits.dim() == 3, "The logits should have dimension with with size (n_samples, n_data, n_bins)"
        G = torch.tril(torch.ones(logits.shape[2], logits.shape[2])).to(logits.device)
        density = torch.softmax(logits, dim=2)
        G_with_samples = G.expand(density.shape[0], -1, -1)

        # b: n_samples; i: n_data; j: n_bin; k: n_bin
        return torch.einsum('bij,bjk->bik', density, G_with_samples)
    else:   # no sampling
        assert logits.dim() == 2, "The logits should have dimension with with size (n_data, n_bins)"
        G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
        density = torch.softmax(logits, dim=1)
        return torch.matmul(density, G)

def make_mtlr_prediction(
        model: mtlr,
        x: torch.Tensor,
        time_bins: NumericArrayLike,
        config: argparse.Namespace
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        survival_curves = mtlr_survival(pred, with_sample=False)

    time_bins = torch.cat([torch.tensor([0]), time_bins], dim=0).to(survival_curves.device)
    return survival_curves, time_bins, survival_curves.unsqueeze(0).repeat(config.n_samples_test, 1, 1)

def train_mtlr_rat_model(
        generators: List[nn.Module],
        encoders: List[nn.Module],
        data_train: pd.DataFrame,
        data_val: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: dotdict,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda"),
        args = None
) -> nn.Module:
    if config.verbose:
        print(f"Reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    train_size = data_train.shape[0]
    val_size = data_val.shape[0]
    
    # Make optimizers
    optimizers = list() # contains K optimizers
    for (gen, enc) in zip(generators, encoders):
        optimizer = get_optimizer([gen, enc], args)
        optimizers.append(optimizer)
    
    #if reset_model:
    #    model.reset_parameters()

    #model = model.to(device)
    #model.train()
    best_val_nll = np.inf
    best_ep = -1
    n_time_bins = len(time_bins)

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    x, y = reformat_survival(data_train, time_bins)
    x_val, y_val = reformat_survival(data_val, time_bins)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=config.batch_size, shuffle=False)
    for i in pbar:
        nll_loss = 0
        step = 0
        
        # Training
        for xi, yi in train_loader:
            for gen in generators:
                gen.train()
            for enc in encoders:
                enc.train()
            
            xi, yi = xi.to(device), yi.to(device)
            xi = xi[:, :, None]
            
            step += 1
            if step % 100 == 0 or args.debug_mode:
                args.gumbel_temprature = max(np.exp((step+1) *-1* args.gumbel_decay), .05)
            
            for optimizer in optimizers:
                optimizer.zero_grad()

            masks = list()
            if args.get_rationales:
                for gen in generators:
                    mask, _ = gen(xi)
                    masks.append(mask)
            else:
                for gen in generators:
                    masks.append(None)
            
            total_loss = []
            features = []
            for k in range(n_time_bins): # for each pair of (enc, gen, mask)
                logits = encoders[k].forward(xi, mask=masks[k])
                
                # change yi so that events are consistent
                yi_clamp = yi.cumsum(dim=1).clamp(max=1)
                
                # convert yi
                yi_long = yi_clamp.long()
                yi_one_hot = torch.nn.functional.one_hot(yi_long, num_classes=2)
                yi_one_hot = yi_one_hot.float()
                yi_one_hot_t = torch.transpose(yi_one_hot, 1, 2)
                
                # calculate BCE
                #loss = mtlr_nll(logits, yi, C1=config.c1, average=False)
                criterion = nn.BCELoss()
                logits_probs = F.softmax(logits, dim=1)
                loss = criterion(logits_probs, yi_one_hot_t[:,:,k])
                
                # calculate selection cost
                if args.get_rationales:
                    selection_cost = generators[k].loss(masks[k], xi)
                    loss += args.selection_lambda * selection_cost
                
                total_loss.append(loss.detach().numpy())
                loss.backward()
                optimizers[k].step()
                
                batch_rationales = learn.get_rationales(masks[k])
                features.append(batch_rationales)
            
            #print(f"{total_loss[0]} - {total_loss[5]} - {total_loss[10]}")
            #nll_loss += (loss / train_size).item()

        # Validation
        valid_rationales = defaultdict(int)
        nll_losses, selection_losses = list(), list()
        for xi, yi in valid_loader:
            for gen in generators:
                gen.eval()
            for enc in encoders:
                enc.eval()
                
            xi, yi = xi.to(device), yi.to(device)
            xi = xi[:, :, None]
            
            masks = list()
            if args.get_rationales:
                for gen in generators:
                    mask, _ = gen(xi)
                    masks.append(mask)
            else:
                for gen in generators:
                    masks.append(None)
            
            batch_val_loss, batch_seleciton_cost = 0, 0
            for k in range(n_time_bins):
                logits = encoders[k].forward(xi, mask=masks[k])
                yi_clamp = yi.cumsum(dim=1).clamp(max=1)
                yi_long = yi_clamp.long()
                yi_one_hot = torch.nn.functional.one_hot(yi_long, num_classes=2)
                yi_one_hot = yi_one_hot.float()
                yi_one_hot_t = torch.transpose(yi_one_hot, 1, 2)
                criterion = nn.BCELoss()
                logits_probs = F.softmax(logits, dim=1)
                val_loss_k = criterion(logits_probs, yi_one_hot_t[:,:,k])
                
                if args.get_rationales:
                    selection_cost = generators[k].loss(masks[k], xi)
                    val_loss_k += args.selection_lambda * selection_cost
                    batch_seleciton_cost += selection_cost.detach().numpy()
                
                batch_val_loss += val_loss_k.detach().numpy()
                
                if args.get_rationales:
                    batch_rationales_k = learn.get_rationales(masks[k])
                    for feature_list in batch_rationales_k:
                        for feature in feature_list:
                            valid_rationales[feature] += 1
                
            batch_val_loss /= n_time_bins
            
            if args.get_rationales:
                batch_seleciton_cost /= n_time_bins
            
            nll_losses.append(batch_val_loss)
            
            if args.get_rationales:
                selection_losses.append(batch_seleciton_cost)
        
        if args.get_rationales:
            valid_rationales = dict(valid_rationales)
            valid_rationales = dict(sorted(valid_rationales.items(),
                                           key=lambda item: item[1], reverse=True))
            
        mean_valid_nll = np.mean(nll_losses)
        
        if args.get_rationales:
            mean_valid_selection = np.mean(selection_losses)
        
        pbar.set_description(f"[epoch {k + 1: 4}/{config.num_epochs}]")
        if args.get_rationales:
            pbar.set_postfix_str(f"Valid nll = {mean_valid_nll.item():.4f}; " 
                                 f"Valid selection = {mean_valid_selection:.4f};")
        else:
            pbar.set_postfix_str(f"Valid nll = {mean_valid_nll.item():.4f};")
        if config.early_stop:
            if best_val_nll > mean_valid_nll:
                best_val_nll = mean_valid_nll
                best_ep = k
            if (k - best_ep) > config.patience:
                break
            
    end_time = datetime.now()
    training_time = end_time - start_time
    
    return encoders

def train_mtlr_model(
        model: nn.Module,
        data_train: pd.DataFrame,
        data_val: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: dotdict,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    train_size = data_train.shape[0]
    val_size = data_val.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    x, y = reformat_survival(data_train, time_bins)
    x_val, y_val = reformat_survival(data_val, time_bins)
    x_val, y_val = x_val.to(device), y_val.to(device)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=config.batch_size, shuffle=True)
    for i in pbar:
        nll_loss = 0
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(xi)
            loss = mtlr_nll(y_pred, yi, model, C1=config.c1, average=False)

            loss.backward()
            optimizer.step()

            nll_loss += (loss / train_size).item()
        logits_outputs = model.forward(x_val)
        eval_nll = mtlr_nll(logits_outputs, y_val, model, C1=0, average=True)
        pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
        pbar.set_postfix_str(f"nll-loss = {nll_loss:.4f}; "
                             f"Validation nll = {eval_nll.item():.4f};")
        if config.early_stop:
            if best_val_nll > eval_nll:
                best_val_nll = eval_nll
                best_ep = i
            if (i - best_ep) > config.patience:
                break
    end_time = datetime.now()
    training_time = end_time - start_time
    # model.eval()
    return model

def mtlr_nll(
        logits: torch.Tensor,
        target: torch.Tensor,
        C1: float,
        average: bool = False
) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    logits : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    target : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the encoded ground truth survival.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.
    average
        Whether to compute the average log likelihood instead of sum
        (useful for minibatch training).

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    censored = target.sum(dim=1) > 1
    nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
    nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0

    # the normalising constant
    norm = torch.logsumexp(logits, dim=1).sum()

    nll_total = -(nll_censored + nll_uncensored - norm)
    if average:
        nll_total = nll_total / target.size(0)

    return nll_total

def masked_logsumexp(
        x: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1
) -> torch.Tensor:
    """Computes logsumexp over elements of a tensor specified by a mask
    in a numerically stable way.

    Parameters
    ----------
    x
        The input tensor.
    mask
        A tensor with the same shape as `x` with 1s in positions that should
        be used for logsumexp computation and 0s everywhere else.
    dim
        The dimension of `x` over which logsumexp is computed. Default -1 uses
        the last dimension.

    Returns
    -------
    torch.Tensor
        Tensor containing the logsumexp of each row of `x` over `dim`.
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return torch.log(
        torch.sum(torch.exp((x - max_val.unsqueeze(dim)) * mask) * mask,
                  dim=dim)) + max_val
