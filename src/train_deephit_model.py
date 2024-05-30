import pandas as pd
import numpy as np
import config as cfg
import torch
import random
import warnings
from utility import dotdict, preprocess_data, make_time_bins, make_stratified_split, convert_to_structured, split_time_event
from data_loader import SyntheticDataLoader
from sksurv.metrics import concordance_index_censored
import numpy as np
import matplotlib.pyplot as plt
import torchtuples as tt
from deephit.models.deephit import DeepHitSingle
import math

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

if __name__ == "__main__":
    # Load data
    dl = SyntheticDataLoader().load_data()
    num_features, cat_features = dl.get_features()
    df = dl.get_data().astype('float32')
    
    # Split data
    df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                        frac_valid=0.1, frac_test=0.2, random_state=0)
    X_train = df_train[cat_features+num_features]
    X_valid = df_valid[cat_features+num_features]
    X_test = df_test[cat_features+num_features]
    
    # Scale data
    X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test, cat_features,
                                               num_features, as_array=True)
    
    # Make discrete
    num_durations = math.ceil(math.sqrt(len(df['time'].values)))
    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df['time'].values, df['event'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_valid = labtrans.transform(*get_target(df_valid))

    valid = (X_valid, y_valid)

    # We don't need to transform the test labels
    durations_test, events_test = get_target(df_test)
    
    in_features = X_train.shape[1]
    num_nodes = [32, 32]
    out_features = labtrans.out_features
    batch_norm = True
    dropout = 0.1

    # Build MLP
    net = torch.nn.Sequential(
        torch.nn.Linear(in_features, 32),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(32),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(32),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(32, out_features))
    
    # Train Deephit
    model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
    model.optimizer.set_lr(0.01)
    
    epochs = 100
    batch_size = 32
    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(X_train, y_train, batch_size, epochs,
                    callbacks, val_data=(X_valid, y_valid))
    
    _ = log.plot()
    
    # Make predictions
    surv = model.predict_surv_df(X_test)
    print(surv.T)

