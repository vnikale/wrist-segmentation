import numpy as np
import tensorflow as tf
from scipy.io import savemat
import albumentations as albu
import os
from tqdm import tqdm
import cv2

def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

def make_rgb(image):
    img_rgb = np.copy(image)
    img_rgb = np.append(img_rgb,image,axis=3)
    img_rgb = np.append(img_rgb,image,axis=3)

    img_rgb_normalized = img_rgb - np.min(img_rgb)
    img_rgb_normalized /= np.max(img_rgb) - np.min(img_rgb)
    # img_rgb_normalized = img_as_ubyte(img_rgb_normalized)

    return img_rgb_normalized

def aug_transforms():
    return [
        albu.VerticalFlip(p=0.7),
        albu.HorizontalFlip(p=0.7),
        albu.Rotate(limit=180, interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_CONSTANT, value=None,
                    mask_value=None, always_apply=False, p=0.6),
        albu.ElasticTransform(alpha=10, sigma=30, alpha_affine=25,
                              interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_WRAP, value=None,
                              mask_value=None, always_apply=False, approximate=False, p=0.4),
        albu.GridDistortion(num_steps=15, distort_limit=0.2, interpolation=cv2.INTER_LANCZOS4,
                            border_mode=cv2.BORDER_WRAP, value=None, mask_value=None,
                            always_apply=False, p=0.4)
    ]


def augment(X, y, factor=10, fold='Augmented', pat_len=None):
    transforms = albu.Compose(aug_transforms())

    set_imag = X
    set_mask = y

    N = set_imag.shape[0]

    aug_size = list(set_imag.shape)
    aug_size[0] = N * factor
    print(aug_size)

    fold = fold + str(factor) + '\\'
    if not os.path.exists(fold):
        os.makedirs(fold)

    if pat_len is None:
        with tqdm(total=aug_size[0]) as pbar:
            for i in range(0, factor):
                for j in range(N):
                    pbar.update(1)
                    if i == 0:
                        mdic = {"original": set_imag[j, :, :, 0], "mask": set_mask[j, :, :, 0]}
                        savemat(fold + 'Slice{0}.mat'.format(i * N + j), mdic)
                    else:
                        a = transforms(image=set_imag[j, :, :, 0], mask=set_mask[j, :, :, 0])
                        #         aug_imag[i*N_train + j, ..., 0] = a['image']/np.max(a['image'])
                        #         aug_mask[i*N_train + j, ..., 0] = a['mask']
                        mdic = {"original": a['image'] / np.max(a['image']), "mask": a['mask']}
                        savemat(fold + 'Slice{0}.mat'.format(i * N + j), mdic)
    else:
        with tqdm(total=aug_size[0]) as pbar:
            summ = np.insert(np.cumsum(pat_len),0,0)
            for i in range(0, factor):
                for k in range(len(pat_len)):
                    path = os.path.join(fold, 'Patient%03d' % int(k + 1))
                    if not os.path.exists(path):
                        os.makedirs(path)
                    for j in range(pat_len[k]):
                        pbar.update(1)
                        nn = pat_len[k]
                        if i == 0:
                            mdic = {"original": set_imag[summ[k] + j, :, :, 0], "mask": set_mask[summ[k] + j, :, :, 0]}
                            savemat(os.path.join(path, 'Slice{0}.mat'.format(i * nn + j)), mdic)
                        else:
                            a = transforms(image=set_imag[summ[k] + j, :, :, 0], mask=set_mask[summ[k] + j, :, :, 0])
                            mdic = {"original": a['image'] / np.max(a['image']), "mask": a['mask']}
                            savemat(os.path.join(path, 'Slice{0}.mat'.format(i * nn + j)), mdic)