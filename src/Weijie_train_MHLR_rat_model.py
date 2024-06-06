import pandas as pd
import numpy as np
import config as cfg
import random
import warnings
from mtlr import Encoder, train_mtlr_model, make_mtlr_prediction, get_optimizer, mtlr_nll, mtlr_survival #train_mtlr_rat_model
from utility import dotdict, preprocess_data, make_time_bins, make_stratified_split, convert_to_structured, split_time_event
from data_loader import SyntheticDataLoader
from sksurv.metrics import concordance_index_censored
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from typing import List, Tuple, Optional, Union
from datetime import datetime
import torch.optim as optim
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
from mtlr import Encoder, train_mtlr_model, make_mtlr_prediction, get_optimizer #train_mtlr_rat_model


# Rational extration
from rational.generator import Generator

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)


from mhlr_rat_model import MHLR_rational_Model

def train_mhlr_rat_model(
        model,
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
    val_size = data_var.shape[0]

    # Make optimizers
    optimizer = optim.Adam(model.parameters(), lr=config.lr)


    # if reset_model:
    #     model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1
    n_time_bins = len(time_bins)

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    x, y = reformat_survival(data_train, time_bins)
    x_val, y_val = reformat_survival(data_var, time_bins)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=config.batch_size, shuffle=False)
    for i in pbar:
        nll_loss = 0
        step = 0

        # Training
        for xi, yi in train_loader:
            xi = xi[:, :, None]
            output = model(xi)
            loss = mtlr_nll(output, yi, C1=0.001, average=False)
            loss.backward()
            optimizer.step()
            #print(f"{total_loss[0]} - {total_loss[5]} - {total_loss[10]}")
            # nll_loss += (loss / train_size).item()

        # Validation
        valid_rationales = defaultdict(int)
        nll_losses, selection_losses = list(), list()
        for xi, yi in valid_loader:
            xi = xi[:, :, None]
            output = model(xi)
            loss = mtlr_nll(output, yi, C1=0.001, average=False)        
            batch_val_loss = loss.item()
    #         if args.get_rationales:
    #             selection_cost = generators[k].loss(masks[k], xi)
    #             val_loss_k += args.selection_lambda * selection_cost
    #             batch_seleciton_cost += selection_cost.detach().numpy()                
    #             batch_rationales_k = learn.get_rationales(masks[k])
    #             for feature_list in batch_rationales_k:
    #                 for feature in feature_list:
    #                     valid_rationales[feature] += 1

    #             batch_val_loss += val_loss_k.detach().numpy()
    #             batch_seleciton_cost /= n_time_bins
    #             selection_losses.append(batch_seleciton_cost)


            # batch_val_loss /= n_time_bins
            nll_losses.append(batch_val_loss)


    #     if args.get_rationales:
    #         valid_rationales = dict(valid_rationales)
    #         valid_rationales = dict(sorted(valid_rationales.items(),
    #                                        key=lambda item: item[1], reverse=True))

    #         mean_valid_selection = np.mean(selection_losses)


        mean_valid_nll = np.mean(nll_losses)

        pbar.set_description(f"[epoch {i: 4}/{config.num_epochs}]")
        if args.get_rationales:
            pbar.set_postfix_str(f"Valid nll = {mean_valid_nll.item():.4f}; " )
                                 # f"Valid selection = {mean_valid_selection:.4f};")
        else:
            pbar.set_postfix_str(f"Valid nll = {mean_valid_nll.item():.4f};")
        if config.early_stop:
            if best_val_nll > mean_valid_nll:
                best_val_nll = mean_valid_nll
                best_ep = i
            if (i - best_ep) > config.patience:
                break

    end_time = datetime.now()
    training_time = end_time - start_time
    return model

# Load data
dl = SyntheticDataLoader().load_data()
num_features, cat_features = dl.get_features()
df = dl.get_data()

# Split data
df_train, df_var, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                    frac_valid=0.1, frac_test=0.2, random_state=0)
X_train = df_train[cat_features+num_features]
X_var = df_var[cat_features+num_features]
X_test = df_test[cat_features+num_features]
y_train = convert_to_structured(df_train["time"], df_train["event"])
y_var = convert_to_structured(df_var["time"], df_var["event"])
y_test = convert_to_structured(df_test["time"], df_test["event"])

# Scale data
X_train, X_var, X_test = preprocess_data(X_train, X_var, X_test, cat_features, num_features)

# Make time/event split
t_train, e_train = split_time_event(y_train)
t_var, e_var = split_time_event(y_var)
t_test, e_test = split_time_event(y_test)

# Make event times
time_bins = make_time_bins(t_train, event=e_train)

# Format data
data_train = X_train.copy()
data_train["time"] = pd.Series(y_train['time'])
data_train["event"] = pd.Series(y_train['event']).astype(int)
data_var = X_var.copy()
data_var["time"] = pd.Series(y_var['time'])
data_var["event"] = pd.Series(y_var['event']).astype(int)
data_test = X_test.copy()
data_test["time"] = pd.Series(y_test['time'])
data_test["event"] = pd.Series(y_test['event']).astype(int)
num_features = X_train.shape[1]

# Make generators
args = dotdict(cfg.PARAMS_RATIONAL)
args['dropout'] = 0.25
n_intervals = len(time_bins)
config = dotdict(cfg.PARAMS_MTLR)

from mhlr_rat_model import MHLR_rational_Model
num_time_bins = len(time_bins)+1

model = MHLR_rational_Model(5, num_time_bins, args)
# model
random_state = 0
reset_model=True
config.early_stop = False

model = train_mhlr_rat_model(model, data_train, data_var, time_bins,
                                            config, random_state=0, reset_model=True, device=device, args=args)

x_test = torch.tensor(data_test.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
x_test = x_test[:, :, None]

surv_preds, _, _ = make_mtlr_prediction(model, x_test, time_bins, config)
survival_prob = surv_preds.T.detach().numpy()

time_bins = torch.cat([torch.tensor([0]).to(time_bins.device), time_bins])
survival_curves = pd.DataFrame.from_records(survival_prob, 
                         index=np.array(time_bins))

from evaluation import ISD_evaluation # check the path

ISD_eval_dict = ISD_evaluation(survival_curves, data_test[['time', 'event']], data_test[['time', 'event']], verbose=False)
ISD_eval_dict['d_cal'] = ISD_eval_dict['d_cal'][0] >= 0.05

print (ISD_eval_dict)