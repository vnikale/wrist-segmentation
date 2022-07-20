from natsort import natsorted
import glob
import re
import numpy as np
import os
from tqdm import tqdm

from ..preprocess import preprocess_wrapper

from scipy.io import loadmat
import nibabel as nib


def load_mat(file: str):
    '''
    Load .mat files of two types which was used initially in the work
    :param file: path to the file
    :return: image and corresponding mask from the .mat file
    '''
    assert os.path.splitext(file)[-1] == '.mat', 'Error: file is not .mat'

    slic = loadmat(file)

    if 'Slice' in slic.keys():
        slic = slic['Slice']
        image = slic[..., 0]
        mask = slic[..., 1]
    else:
        image = slic['original']
        mask = slic['mask']

    return image, mask


class TrainTestDataloaderMat():
    '''
    Full dataset loader from folder to the memory with transformation using functions from the preprocess module.
    This loader required both image and mask
    Dataset directory must follow the next structure:
    ./data/
        ./data/(word)(1-7 digits)
    e.g.
    ./data/
        ./data/Patient001
            ./data/Patient001/... (.mat/.nii)
            ...
        ./data/Subj01
        ...

    '''
    def __init__(self, folder, pre_args, config, limit = None, pattern = r'\w(\d{1,7})'):
        self.folder = folder
        self.pre_args = pre_args
        self.config = config
        self.limit = limit
        self.pattern = pattern

    def load(self):

        paths = glob.glob(self.folder)
        paths = natsorted(paths)

        if self.limit is not None:
            ind = np.random.randint(0, len(paths), size=self.limit, dtype=int)
            paths = np.array(paths)[ind]

        subjects = list(set(re.findall(self.pattern, str('').join([os.path.dirname(path) for path in paths]))))
        N_pat = np.array([int(subj) for subj in subjects])
        N_pat.sort()
        N_pat = N_pat[-1]

        X = np.zeros((len(paths), *self.config.IMAGE_SIZE, 1))
        y = np.zeros((len(paths), *self.config.IMAGE_SIZE, 1))
        subj_len = np.zeros((N_pat,), dtype=np.int)

        i = 0
        for ID in tqdm(paths):
            image, mask = load_mat(ID)

            self.pre_args['image'] = image
            self.pre_args['mask'] = mask

            image, mask = preprocess_wrapper(self.pre_args)

            X[i, ..., 0] = image
            y[i, ..., 0] = mask

            N = int(re.findall(self.pattern, str('').join(os.path.dirname(ID)))[0])
            subj_len[N - 1] += 1
            i += 1

        return X, y, subj_len

class TrainTestDataloaderNii():
    '''
    Full dataset loader from folder to the memory with transformation using functions from the preprocess module.


    '''
    def __init__(self, folder, pre_args, config, limit = None, pattern = r'\w(\d{1,7})'):
        self.folder = folder
        self.pre_args = pre_args
        self.config = config
        self.limit = limit
        self.pattern = pattern

    def load(self):
        paths = glob.glob(self.folder)
        paths = natsorted(paths)

        if self.limit is not None:
            ind = np.random.randint(0, len(paths), size=self.limit, dtype=int)
            paths = np.array(paths)[ind]

        subjects = list(set(re.findall(self.pattern, str('').join(paths))))
        subjects = [int(subj) for subj in subjects]
        subjects.sort()
        N_pat = len(subjects)
        map_subj = dict(zip(subjects,np.arange(0,N_pat)))

        subj_len = np.zeros((N_pat,), dtype=np.int)

        patterns = ['pc\\pc0.*', 'pc\\pc.*', 'qdess\echo1*', 'qdess\echo2*']
        names = ['mask0', 'mask1', 'image0', 'image1']
        dataset = {'image0':[], 'image1':[], 'mask0':[], 'mask1':[]}

        data = {}
        i = 0
        for ID in tqdm(paths):
            for i, pat in enumerate(patterns):
                file = glob.glob(os.path.join(ID,pat))[0]
                data[names[i]] = nib.load(file).get_fdata()

            inputs = {}
            inputs['data'] = data
            self.pre_args['inputs'] = inputs
            outputs = preprocess_wrapper(self.pre_args)

            for name in names:
                dataset[name].append(outputs['data'][name])

            N = int(re.findall(self.pattern, str('').join(ID))[0])
            N = map_subj[N]
            subj_len[N] = outputs['data'][name].shape[0]
            i += 1

        # TODO: make it faster, now it is very slow
        for name in names:
            dataset[name] = np.array(dataset[name]).reshape((-1, *self.config.IMAGE_SIZE))

        return dataset, subj_len



