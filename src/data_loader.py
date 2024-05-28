import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, List
from pathlib import Path
import config as cfg
from utility import convert_to_structured
import copy

class BaseDataLoader(ABC):
    """
    Base class for data loaders.
    """
    def __init__(self):
        """Initilizer method that takes a file path, file name,
        settings and optionally a converter"""
        self.X: pd.DataFrame = None
        self.y: np.ndarray = None
        self.num_features: List[str] = None
        self.cat_features: List[str] = None

    @abstractmethod
    def load_data(self, n_samples) -> None:
        """Loads the data from a data set at startup"""
        
    def get_data(self) -> pd.DataFrame:
        """
        This method returns the features and targets
        :return: df
        """
        df = pd.DataFrame(self.X)
        df['time'] = self.y['time']
        df['event'] = self.y['event']
        return df

    def get_features(self) -> List[str]:
        """
        This method returns the names of numerical and categorial features
        :return: the columns of X as a list
        """
        return self.num_features, self.cat_features

    def _get_num_features(self, data) -> List[str]:
        return data.select_dtypes(include=np.number).columns.tolist()

    def _get_cat_features(self, data) -> List[str]:
        return data.select_dtypes(['object']).columns.tolist()

class SyntheticDataLoader(BaseDataLoader):
    """
    Data loader for synthetic data
    """
    def load_data(self):
        params = cfg.SYNTHETIC_SETTINGS
        raw_data, event_times, labels = self.make_synthetic()
        if params['discrete'] == False:
            min_time = np.min(event_times[event_times != -1]) 
            max_time = np.max(event_times[event_times != -1]) 
            time_range = max_time - min_time
            bin_size = time_range / params['num_bins']
            binned_event_time = np.floor((event_times - min_time) / bin_size)
            binned_event_time[binned_event_time == params['num_bins']] = params['num_bins'] - 1 
        columns = [f'x{num}' for num in range(1, raw_data.shape[1]+1)]
        self.X = pd.DataFrame(raw_data, columns=columns)
        self.y = convert_to_structured(binned_event_time, labels)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self
    
    '''
    Generate synthetic dataset based on:
    https://github.com/MLD3/Hierarchical_Survival_Analysis
    '''
    def make_synthetic(self):
        num_event = 1
        num_data = 1000 # def. 1000
        num_feat = 2 #in each segment, total = 15 (5 features x 3 segments) # def. 5.
        
        #construct covariates
        bounds = np.array([-5, -10, 5, 10])
        x_11 = np.random.uniform(bounds[0], bounds[2], size=(num_data//2, num_feat))
        x_12 = np.random.uniform(bounds[0], bounds[2], size=(num_data//2, num_feat))
        x_21 = np.random.uniform(bounds[1], bounds[3], size=(num_data//2, num_feat))
        x_31 = np.random.uniform(bounds[1], bounds[3], size=(num_data//2, num_feat)) 
        x_22 = np.random.uniform(bounds[1], bounds[3], size=(num_data//2, num_feat))
        x_32 = np.random.uniform(bounds[1], bounds[3], size=(num_data//2, num_feat)) 
        
        x1 = np.concatenate((x_11, x_21, x_31), axis=1)
        x2 = np.concatenate((x_12, x_32, x_22), axis=1)
        x = np.concatenate((x1, x2), axis=0)
        
        #construct time to events
        gamma_components = []
        gamma_const = [1, 1, 1]
        for i in range(num_event + 1):
            gamma_components.append(gamma_const[i] * np.ones((num_feat,)))
        gamma_components.append(gamma_const[-1] * np.ones((num_feat,)))

        distr_noise = 0.4
        
        event_times = [] 
        raw_event_times = []
        for i in range(num_event):
            raw_time = np.power(np.matmul(np.power(np.absolute(x[:, :num_feat]), 1), gamma_components[0]), 2) + \
                    np.power(np.matmul(np.power(np.absolute(x[:, (i + 1)*num_feat:(i+2)*num_feat]), 1), gamma_components[i + 1]), 2)
            raw_event_times.append(raw_time)
            times = np.zeros(raw_time.shape)
            for j in range(raw_time.shape[0]):
                times[j] = np.random.lognormal(mean=np.log(raw_time[j]), sigma=distr_noise)
            event_times.append(times)

        t = np.zeros((num_data, num_event))
        for i in range(num_event):
            t[:, i] = event_times[i]
        labels = np.ones(t.shape)
        
        #enforce a prediction horizon
        horizon = np.percentile(np.min(t, axis=1), 50) 
        for i in range(t.shape[1]):
            censored = np.where(t[:, i] > horizon)
            t[censored, i] = horizon
            labels[censored, i] = 0
        
        print('label distribution: ', np.unique(labels, return_counts=True, axis=0))
        return x, t, labels