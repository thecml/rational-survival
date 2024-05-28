from pathlib import Path
import numpy as np

# Directories
ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = Path.joinpath(ROOT_DIR, "data")
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
CONFIGS_DIR = Path.joinpath(ROOT_DIR, 'configs')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')
MTLR_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'mtlr')

'''
Record of all settings for datasets
Definitions:
    location: where the data is stored
    features: where in location (see above) the covariates/features are stored
    terminal event: event such that no other events can occur after it
    discrete: whether the time values are discrete
    event ranks: each key represents and event, the values are the events that prevent it
    event groups: each key represents the position in a trajectory (e.g., first, second, ...), values represent which events can occur in that position
    min_time: earliest event time
    max_time: latest event time (prediction horizon)
    min_epoch: minimum number of epochs to train for (while learning the model)
'''
SYNTHETIC_SETTINGS = {
    'num_events': 1,
    'num_bins': 20,
    'terminal_events': [1],
    'discrete': False,
    'event_ranks': {0:[], 1:[]},
    'event_groups': {0:[0, 1], 1:[0, 1]},
    'min_time': 0,
    'max_time': 20,
    'min_epoch': 50,
}

PARAMS_MTLR = {
    'hidden_size': 32,
    'verbose': False,
    'lr': 0.00008,
    'c1': 0.01,
    'num_epochs': 100,
    'dropout': 0.5,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 10
}