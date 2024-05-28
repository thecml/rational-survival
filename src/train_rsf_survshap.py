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
from sksurv.ensemble import RandomSurvivalForest
from survshap import SurvivalModelExplainer, PredictSurvSHAP, ModelSurvSHAP

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
    
    # prepare survival model
    model = RandomSurvivalForest()
    model.fit(X_train, y_train)

    # create explainer
    explainer = SurvivalModelExplainer(model=model, data=X_train, y=y_train)

    # compute SHAP values for a single instance
    observation_A = X_test.iloc[[0]]
    survshap_A = PredictSurvSHAP()
    survshap_A.fit(explainer=explainer, new_observation=observation_A)

    survshap_A.result
    survshap_A.plot()

    # compute SHAP values for a group of instances
    model_survshap = ModelSurvSHAP(calculation_method="treeshap") # fast implementation for tree-based models
    model_survshap.fit(explainer=explainer, new_observations=observation_A)

    model_survshap.result
    model_survshap.plot_mean_abs_shap_values()
    model_survshap.plot_shap_lines_for_all_individuals(variable="x1")
    extracted_survshap = model_survshap.individual_explanations[0] # PredictSurvSHAP object
    