from natsort import natsorted
import glob
import re
import numpy as np
import os
from tqdm import tqdm
from .preprocess import preprocess_wrapper

class TrainTestDataloader():
    '''
    Full dataset loader from folder to the memory with transformation using functions from the preprocess module.
    This loader required both image and mask
    '''
    def __init__(self, folder, pre_args, load_func, config, limit = False):
        self.folder = folder
        self.pre_args = pre_args
        self.load_func = load_func
        self.config = config
        self.limit = limit

    def load(self):
        paths = glob.glob(self.folder)
        paths = natsorted(paths)

        subjects = list(set(re.findall(r'\w(\d{1,5})', str('').join([os.path.dirname(path) for path in paths]))))
        N_pat = np.array([int(subj) for subj in subjects])
        N_pat.sort()
        N_pat = N_pat[-1]


        X = np.zeros((len(paths), *self.config.IMAGE_SIZE, 1))
        y = np.zeros((len(paths), *self.config.IMAGE_SIZE, 1))
        subj_len = np.zeros((N_pat,), dtype=np.int)

        i = 0
        for ID in tqdm(paths):
            image, mask = self.load_func(ID)
            self.pre_args['image'] = image
            self.pre_args['mask'] = mask

            image, mask = preprocess_wrapper(self.pre_args)

            X[i, ..., 0] = image
            y[i, ..., 0] = mask

            N = int(re.findall(r'\w(\d{1,5})', str('').join(os.path.dirname(ID)))[0])
            subj_len[N - 1] += 1
            i += 1

        return X, y, subj_len



