from tensorflow.keras.callbacks import Callback
import numpy as np
import tensorflow as tf
from .metrics import dice_coef
from datetime import datetime
import os
from .other import make_rgb

def schedule(lr0):
    def sch(epoch, lr):
        if epoch%20==0:
            return lr0
        else:
            return lr * np.math.exp(-0.09)
    return sch

def gencallbacks(config):
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss', mode='min', min_delta=0),
                 tf.keras.callbacks.ModelCheckpoint(
                     filepath=os.path.join(config.logdir,
                                           config.MODEL_NAME + r'_model.{epoch:02d}-{val_dice_coef:.4f}-{val_loss:.6f}_'
                     + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5'),
                     monitor='val_dice_coef',
                     verbose=1,
                     save_best_only=True,
                     mode='max'),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(config.logdir,
                                                            config.MODEL_NAME + r"_log_" \
                                               + datetime.now().strftime("%Y%m%d-%H%M%S")))
    ]
    if config.SCHEDULER == 'exp_restart':
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(schedule(config.LR),verbose=1))

    return callbacks

class TensorBoardImage(Callback):
    def __init__(self, log_dir, model, x, y, freq=1):
        assert isinstance(x, (np.ndarray, np.generic) ) and isinstance(y, (np.ndarray, np.generic) ), \
        'X, y must be numpy arrays'

        super().__init__()
        self.model = model
        self.log_dir = log_dir
        self.freq = freq
        self.x, self.y = x, y
        self.writer = tf.summary.create_file_writer(self.log_dir, filename_suffix='images')
        self.writer_dsc = tf.summary.create_file_writer(self.log_dir+'_DSC', filename_suffix='dsc')

    def on_train_end(self, _):
        self.writer.close()
        self.writer_dsc.close()

    def write_image(self, image, name, tag, epoch):
        with self.writer.as_default():
            tf.summary.image(name, image, step=epoch,description=tag)
    def write_dsc(self, dsc, name, epoch):
        with self.writer_dsc.as_default():
            tf.summary.scalar(name, dsc, step=epoch)
    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1)%self.freq == 0:
            if len(np.squeeze(self.x).shape) > 2:
                n = self.x.shape[0]
                if n > 40:
                    n = 40
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

            self.write_image(image1, 'Ground-Truth', "Ground-Truth", epoch+1)

            ima = np.append(image0, image0, axis=0)
            ima = np.append(ima, ima, axis=0)

            pre = self.model.predict(ima)
            pred0 = pre[0,...]
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
            self.write_image(image2, "Predicted",
                             'Predicted, DSC {0}'.format(dsc),
                             epoch+1)
            self.write_dsc(dsc,'2D DSC',epoch+1)