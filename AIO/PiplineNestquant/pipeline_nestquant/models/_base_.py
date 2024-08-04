import os
from abc import abstractmethod
import json
import time

from utils.visualize import save_plot
from utils.metrics import score

from utils.general import convert_seconds
import numpy as np
import pickle
from utils.general import yaml_load
from pathlib import Path
from utils.npy_utils import NpyFileAppend


from rich.progress import track
from rich.progress import Progress
from rich.progress import BarColumn 
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import MofNCompleteColumn
from rich.progress import TimeRemainingColumn

class BaseModel:
    def __init__(self, modelConfigs, save_dir='.'):
        self.history = None
        self.time_used = '0s'
        self.model = None
        self.modelConfigs = yaml_load(modelConfigs)

        self.dir_log          = 'logs'
        self.dir_plot         = 'plots'
        self.dir_value        = 'values'
        self.dir_model        = 'models'
        self.dir_weight       = 'weights'
        self.dir_architecture = 'architectures'
        self.mkdirs(path=save_dir)

    def mkdirs(self, path):
        path = Path(path)
        self.path_log          = path / self.dir_log
        self.path_plot         = path / self.dir_plot
        self.path_value        = path / self.dir_value
        self.path_model        = path / self.dir_model
        self.path_weight       = path / self.dir_weight
        self.path_architecture = path / self.dir_architecture

        for p in [self.path_log, self.path_plot, self.path_value, self.path_model, self.path_weight, self.path_architecture]: 
            p.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def build(self, *inputs):
        raise NotImplementedError 

    @abstractmethod
    def preprocessing(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def fit(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def save(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def load(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *inputs):
        raise NotImplementedError

    def plot(self, save_dir, y, yhat, dataset):
        try:
            save_plot(filename=os.path.join(self.path_plot, f'{self.__class__.__name__}-{dataset}.png'),
                      data=[{'data': [range(len(y)), y],
                             'color': 'green',
                             'label': 'y'},
                            {'data': [range(len(yhat)), yhat],
                             'color': 'red',
                             'label': 'yhat'}],
                      xlabel='Sample',
                      ylabel='Value')
        except: pass

    def score(self, 
              y, 
              yhat, 
              # path=None,
              r):
        return score(y=y, 
                     yhat=yhat, 
                     # path=path, 
                     # model=self.__class__.__name__,
                     r=r)

    def ProgressBar(self):
        return Progress("[bright_cyan][progress.description]{task.description}",
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TextColumn("•Items"),
                        MofNCompleteColumn(), # "{task.completed}/{task.total}",
                        TextColumn("•Remaining"),
                        TimeRemainingColumn(),
                        TextColumn("•Total"),
                        TimeElapsedColumn())

class MachineLearningModel(BaseModel):
    def __init__(self, modelConfigs, save_dir, **kwargs):
        super().__init__(modelConfigs=modelConfigs, save_dir=save_dir)
        self.is_classifier = False
    
    def build(self): pass

    def preprocessing(self, x, classifier=False):
        if classifier: res = [i.flatten().astype(int) for i in x]
        else: res = [i.flatten() for i in x]

        return res

    def fit(self, X_train, y_train, **kwargs):
        start = time.time()

        y_train = np.ravel(self.preprocessing(y_train, self.is_classifier), order='C') 
        x_train = self.preprocessing(X_train)

        self.model.fit(X=x_train, 
                       y=y_train)
        self.time_used = convert_seconds(time.time() - start)
    
    def save(self, file_name:str, extension:str='.pkl'):
        file_path = Path(self.path_weight, f'{file_name}{extension}')
        pickle.dump(self.model, open(Path(file_path).absolute(), "wb"))
        return file_path

    def load(self, weight):
        if not os.path.exists(weight): pass
        self.model = pickle.load(open(weight, "rb"))

    def predict(self, X, save=True, scaler=None):
        yhat = self.model.predict(self.preprocessing(x=X))
        if scaler is not None: 
            yhat = yhat * (scaler['max'] - scaler['min']) +  scaler['min']
            
        if save: 
            filename = self.path_value / f'yhat-{self.__class__.__name__}.npy'
            np.save(file=filename, 
                    arr=yhat, 
                    allow_pickle=True, 
                    fix_imports=True)
        return yhat
