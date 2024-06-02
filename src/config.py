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
    'verbose': True,
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

PARAMS_RATIONAL = {
    'train': True,
    'test': True,
    'cuda': False,
    'num_gpus': 1,
    'debug_mode': False,
    'class_balance': False,
    'objective': 'cross_entropy',
    'aspect': 'overall',
    'init_lr': 0.001,
    'epochs': 20,
    'batch_size': 32,
    'patience': 10,
    'tuning_metric': 'loss',
    'save_dir': 'snapshot',
    'results_path': 'results',
    'snapshot': None,
    'num_workers': 4,
    'model_form': 'mlp',
    'hidden_dim': 100,
    'num_layers': 3,
    'dropout': 0.25,
    'weight_decay': 1e-3,
    'dataset': 'news_group',
    'embedding': 'glove',
    'gumbel_temprature': 1,
    'gumbel_decay': 1e-5,
    'get_rationales': True,
    'selection_lambda': 0.01,
    'continuity_lambda': 0.01,
    'num_class': 2,
    'use_as_tagger': False,
    'tag_lambda': 0.5
}
    