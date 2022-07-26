import numpy as np
import tensorflow as tf


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
