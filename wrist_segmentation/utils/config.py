import os
from datetime import datetime
import yaml
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from .metrics import dice_coef

class BaseConfig():
    def __init__(self, yaml_file):
        script_dir = Path(__file__).parents[2]
        self.MYFOLDER = script_dir

        file_path = os.path.join(script_dir, 'configs', yaml_file + '.yaml')
        with open(file_path, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.LOGDIRECTORY = 'logs'
        for k, v in self.config.items():
            setattr(self, k, v)


    def merge(self, cfg):
        for k, v in cfg.__dict__.items():
            setattr(self, k, v)

class Config(BaseConfig):
    '''
    Config class. Config should be defined in the .yaml config file in the 'configs' folder.
    '''
    def __init__(self,yaml_file, log_config=False):
        super().__init__(yaml_file)

        self._check_childrens()

        self._set_functions()

        self.MODEL_NAME_LOG = self.MODEL_NAME + '_' + str(self.N_DOWN_LAYERS) + 'down_' + str(self.N_UP_LAYERS) + 'up_'

        self.logdir = os.path.join(self.MYFOLDER, 'output', self.LOGDIRECTORY, self.MODEL_NAME_LOG)
        self.make_dirs(self.logdir)

        if log_config:
            self.log_config()

    def _check_childrens(self):
        assert 'MODEL_NAME' in self.__dict__.keys(), 'MODEL_NAME should be defined in config.yaml'
        assert 'N_DOWN_LAYERS' in self.__dict__.keys(), 'N_DOWN_LAYERS should be defined in config.yaml'
        assert 'N_UP_LAYERS' in self.__dict__.keys(), 'N_UP_LAYERS should be defined in config.yaml'

    def _set_functions(self):
        metrics = {'dice_coef':
                   dice_coef,}
        optimizers = {'Adam':
                      Adam,}

        if self.METRIC in metrics.keys():
            self.METRIC = metrics[self.METRIC]
        else:
            raise NotImplementedError('Metric is not found')

        if self.OPTIMIZER in optimizers.keys():
            self.OPTIMIZER = optimizers[self.OPTIMIZER]
        else:
            raise NotImplementedError('Optimizer is not found')

    def make_dirs(self,path):
        if not os.path.exists(path):
            os.makedirs(path)

    def log_config(self):
        filename = os.path.join(self.logdir,'config' + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt')
        for member in dir(self):
            if member[0] != '_':
                with open(filename, 'a') as f:
                    f.write(member + ': ' + str(getattr(self, member)) + '\n')