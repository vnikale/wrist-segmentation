import numpy as np
import tensorflow as tf


def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

