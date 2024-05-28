import pandas as pd
import numpy as np
import config as cfg
import torch
import random
import warnings
from mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from utility import dotdict, preprocess_data, make_time_bins, make_stratified_split, convert_to_structured, split_time_event
from data_loader import SyntheticDataLoader
from sksurv.metrics import concordance_index_censored

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
    df = dl.get_data()
    
    # Split data
    df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                        frac_valid=0.1, frac_test=0.2, random_state=0)
    X_train = df_train[cat_features+num_features]
    X_valid = df_valid[cat_features+num_features]
    X_test = df_test[cat_features+num_features]
    y_train = convert_to_structured(df_train["time"], df_train["event"])
    y_valid = convert_to_structured(df_valid["time"], df_valid["event"])
    y_test = convert_to_structured(df_test["time"], df_test["event"])
    
    # Scale data
    X_train, X_valid, X_test = preprocess_data(X_train, X_valid, X_test, cat_features, num_features)

    # Make time/event split
    t_train, e_train = split_time_event(y_train)
    t_valid, e_valid = split_time_event(y_valid)
    t_test, e_test = split_time_event(y_test)
    
    # Make event times
    time_bins = make_time_bins(t_train, event=e_train)

    # Format data
    data_train = X_train.copy()
    data_train["time"] = pd.Series(y_train['time'])
    data_train["event"] = pd.Series(y_train['event']).astype(int)
    data_valid = X_valid.copy()
    data_valid["time"] = pd.Series(y_valid['time'])
    data_valid["event"] = pd.Series(y_valid['event']).astype(int)
    data_test = X_test.copy()
    data_test["time"] = pd.Series(y_test['time'])
    data_test["event"] = pd.Series(y_test['event']).astype(int)
    num_features = X_train.shape[1]
    
    # Load config, train model
    config = dotdict(cfg.PARAMS_MTLR)
    num_time_bins = len(time_bins)
    model = mtlr(in_features=num_features, num_time_bins=num_time_bins, config=config)
    model = train_mtlr_model(model, data_train, data_valid, time_bins,
                             config, random_state=0, reset_model=True, device=device)

    # Make predictions
    x_test = torch.tensor(data_test.drop(["time", "event"], axis=1).values, dtype=torch.float, device=device)
    surv_preds, _, _ = make_mtlr_prediction(model, x_test, time_bins, config)
    time_bins = torch.cat([torch.tensor([0]).to(time_bins.device), time_bins], 0)
    surv_preds = pd.DataFrame(surv_preds, columns=np.array(time_bins))
    print(surv_preds)
