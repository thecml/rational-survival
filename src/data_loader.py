import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, List
from pathlib import Path
import config as cfg
from utility import convert_to_structured
import copy
import os

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
    def load_data(self, data_path = "data.csv"):
        file_path = os.path.join(cfg.DATA_DIR, data_path)
        raw_data = pd.read_csv(file_path)
        self.X = raw_data.drop(['id', 'eventtime', 'status'], axis=1)
        self.y = convert_to_structured(raw_data['eventtime'], raw_data['status'])
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        return self