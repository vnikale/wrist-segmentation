import wrist_segmentation.data.preprocess as preprocess
import wrist_segmentation.data.loader as loader
from wrist_segmentation.utils.config import Config, BaseConfig
from wrist_segmentation.models import BaseModel
from wrist_segmentation.utils.callbacks import TensorBoardImage, gencallbacks
from wrist_segmentation.utils.other import set_seed

import tensorflow as tf
import os


def run_train(cfg):
    def change_progress(subject, netn, status, nets=4):
        prg = str(int(100 // nets * netn))
        msg = subject + ' ' + status + ' '
        cmd = r'Job modify %CCP_JOBID% /progress:' + prg + r' /progressmsg:"' + msg + '"'
        os.system(cmd)

    configs = []
    for cfg_model in cfg.MODEL_CONFIGS:
        config = Config(cfg_model, log_config=cfg.LOG_CONFIG)
        config.merge(cfg)
        configs.append(config)

    set_seed(cfg.SEED)

    net = 0
    for config in configs:
        folder = config.DATA_PATH

        pre_args = {'preprocess_func': preprocess.full_preprocess,
                   'size': config.IMAGE_SIZE}

        args = {'folder': folder,
               'pre_args': pre_args,
                'limit':2,
                'config': cfg,
               }

        loader_ = loader.TrainTestDataloaderMat(**args)

        X, y, subj_len = loader_.load()
        callbacks = gencallbacks(config)
        model = BaseModel.BaseModel(config)
        model.train(train=(X,y), callbacks=callbacks)

        if cfg.TESLA_PROGRESS:
            change_progress(config.MODEL_NAME_LOG, net, 'is trained', nets=len(configs))
        net += 1
    if cfg.TESLA_PROGRESS:
        os.system(r'Job modify %CCP_JOBID% /progress:' + str(100))

def run_crossvalidation(cfg):
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    import numpy as np
    from datetime import datetime

    def change_progress(subject, kfn, netn, status, kfolds=5, nets=4):
        prg = str(int(100 // kfolds // nets * (kfn * (kfolds - 1) + netn)))
        msg = subject + ' ' + status + ' ' + 'Kfold: ' + str(kfn)
        cmd = r'Job modify %CCP_JOBID% /progress:' + prg + r' /progressmsg:"' + msg + '"'
        os.system(cmd)

    def get_folds(idxs, X0, y0):
        X = np.empty((0, *X0.shape[1:]))
        y = np.empty((0, *X0.shape[1:]))
        for i in idxs:
            l = index[i]
            r = index[i + 1]
            X = np.vstack([X, X0[l:r, ...]])
            y = np.vstack([y, y0[l:r, ...]])
        return X, y

    configs = []
    for cfg_model in cfg.MODEL_CONFIGS:
        config = Config(cfg_model, log_config=cfg.LOG_CONFIG)
        config.merge(cfg)
        configs.append(config)

    set_seed(cfg.SEED)

    folder = cfg.DATA_PATH

    pre_args = {'preprocess_func': preprocess.cv_preprocess,
               'size': cfg.IMAGE_SIZE}

    args = {'folder': folder,
           'pre_args': pre_args,
            'limit': 2,
            'config': cfg,
            }

    loader_ = loader.TrainTestDataloaderMat(**args)

    X, y, subj_len = loader_.load()

    info = cfg.INFO_FILE
    group = pd.read_excel(info)['Patient #'].to_numpy()

    K = 5
    group_kfold = GroupKFold(n_splits=K)
    pats = np.arange(0, len(subj_len))
    index = np.insert(np.cumsum(subj_len), 0, 0)

    kf = 0
    for train, test in group_kfold.split(pats, groups=group):
        print(train)
        print(test)
        print('-------')
        print(group[train])
        print(group[test])
        print('-------next--------')

        train_X, train_y = get_folds(train, X, y)
        test_X, test_y = get_folds(test, X, y)

        net = 0
        for config in configs:
            callbacks = gencallbacks(config)
            model = BaseModel.BaseModel(config)
            imagelogger = TensorBoardImage(os.path.join(config.logdir, config.MODEL_NAME + r"_log_images" \
                                           + datetime.now().strftime("%Y%m%d-%H%M%S")), model, test_X, test_y, freq=5)
            callbacks.append(imagelogger)
            model.summary()
            model.train(train=(train_X, train_y), callbacks=callbacks)

            if cfg.TESLA_PROGRESS:
                change_progress(config.MODEL_NAME_LOG, kf, net, 'is trained', kfolds=K, nets=len(configs))
            net += 1
        kf += 1
    if cfg.TESLA_PROGRESS:
        os.system(r'Job modify %CCP_JOBID% /progress:' + str(100))


if __name__ == '__main__':
    config = BaseConfig('config')

    if config.TYPE == 'cv':
        run_crossvalidation(config)
    elif config.TYPE == 'train':
        run_train(config)
    else:
        raise NotImplementedError('Such type of learning process is not supported yet')