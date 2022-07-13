from tensorflow.keras.callbacks import Callback
import numpy as np
import tensorflow as tf
from metrics.losses import dice_coef

def make_rgb(image):
    img_rgb = np.copy(image)
    img_rgb = np.append(img_rgb,image,axis=3)
    img_rgb = np.append(img_rgb,image,axis=3)

    img_rgb_normalized = img_rgb - np.min(img_rgb)
    img_rgb_normalized /= np.max(img_rgb) - np.min(img_rgb)
    # img_rgb_normalized = img_as_ubyte(img_rgb_normalized)

    return img_rgb_normalized

class TensorBoardImage(Callback):
    '''
    Class for a callback saving images to the tensorboard
    '''
    def __init__(self, log_dir, model, x, y, freq=1):
        assert isinstance(x, (np.ndarray, np.generic) ) and isinstance(y, (np.ndarray, np.generic) ), \
        'X, y must be numpy arrays'

        super().__init__()
        self.model = model
        self.log_dir = log_dir
        self.freq = freq
        self.x, self.y = x, y
        self.writer = tf.summary.create_file_writer(self.log_dir, filename_suffix='images')
        self.writer_dsc = tf.summary.create_file_writer(self.log_dir, filename_suffix='dsc')


    def on_train_end(self, _):
        self.writer.close()

    def write_image(self, image, tag, epoch):
        with self.writer.as_default():
            tf.summary.image(tag, image, step=epoch)
    def write_dsc(self, dsc, name, epoch):
        with self.writer_dsc.as_default():
            tf.summary.scalar(name, dsc, step=epoch)
    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1)%self.freq == 0:
            if len(np.squeeze(self.x).shape) > 2:
                n = self.x.shape[0]
                if n > 20:
                    n = 20
                index = np.random.randint(0,n)
                image0 = self.x[index,...]
                mask = self.y[index,...]
            elif len(np.squeeze(self.x).shape) == 2:
                image0 = self.x
                mask = self.y
            else:
                raise RuntimeError('Bad x,y dimension, must be image or batch of images')

            image0 = np.expand_dims(image0, axis=0)
            mask = np.expand_dims(mask, axis=0)

            image0_rgb = make_rgb(image0)
            image1 = np.copy(image0_rgb)
            image1[...,1] = image1[...,1] + mask[...,0]
            image1[image1>1] = 1

            self.write_image(image1, 'Ground-Truth', epoch+1)

            pred0 = self.model.predict(image0)
            pred = np.array(pred0>0.5,dtype=int)
            TP = pred*mask                # test_mask == 1 and tr_pred == 1
            FP = ((mask*2-2)/(-2)) * pred # test_mask == 0 and tr_pred == 1
            FN = ((pred*2-2)/(-2)) * mask # test_mask == 1 0 and tr_pred == 0
            image2 = np.copy(image0_rgb)
            image2[...,1] += np.abs(TP[...,0])
            image2[...,0] += np.abs(FP[...,0])
            image2[...,2] += np.abs(FN[...,0])
            image2[image2>1] = 1
            dsc = dice_coef(mask.astype(float),np.array(pred0,dtype=float))

            self.write_image(image2,
                             'Predicted, DSC {0}'.format(dsc),
                             epoch+1)
            self.write_dsc(dsc,'2D DSC',epoch+1)