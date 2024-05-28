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
from sksurv.linear_model import CoxPHSurvivalAnalysis
from survlime.survlime_explainer import SurvLimeExplainer

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
    
    # Train model    
    model = CoxPHSurvivalAnalysis()
    model.fit(X_train, y_train)
    
    # Use SurvLimeExplainer class to find the feature importance
    training_features = X_train
    training_events = [event for event, _ in y_train]
    training_times = [time for _, time in y_train]
    
    explainer = SurvLimeExplainer(
        training_features=training_features,
        training_events=training_events,
        training_times=training_times,
        model_output_times=model.unique_times_,
    )
    
    # explanation variable will have the computed SurvLIME values
    explanation = explainer.explain_instance(
        data_row=X_test.iloc[0],
        predict_fn=model.predict_cumulative_hazard_function,
        num_samples=500,
    )
    print(explanation)
    
    # Display the weights
    explainer.plot_weights()
