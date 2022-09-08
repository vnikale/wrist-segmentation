from importlib import import_module
import time
import tensorflow as tf

class BaseModel:
    '''
    BaseModel class used in initial work. It was designed to use keras functional api.
    To use class api please create its realization.
    '''
    def __init__(self, config):
        self.config = config
        if config.DISTRUBUTE_TRAIN:
            self.mirrored_strategy = tf.distribute.MirroredStrategy(
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

        #Generate the model
        try:
            model_name = config.MODEL_NAME
            print('Importing {0:s}'.format(model_name))
            module = import_module('wrist_segmentation.models.' + model_name)
            model_gen = module.model_gen
        except:
            print('Error: module {0:s} is not exist!'.format(model_name))

        self.iscompiled = False
        if config.DISTRUBUTE_TRAIN:
            with self.mirrored_strategy.scope():
                self.model = model_gen(config)
        else:
            self.model = model_gen(config)

        if 'WEIGHTS_PATH' in config.__dict__.keys() and 'FINETUNE' in config.__dict__.keys():
            if config.FINETUNE:
                if config.DISTRUBUTE_TRAIN:
                    with self.mirrored_strategy.scope():
                        self.compile()
                        self.model.load_weights(config.WEIGHTS_PATH)
                else:
                    self.compile()
                    self.model.load_weights(config.WEIGHTS_PATH)

                print(f'Finetuning, loaded {config.WEIGHTS_PATH}')


    def summary(self):
        self.model.summary(line_length=120)

    def compile(self):
        if self.config.DISTRUBUTE_TRAIN:
            with self.mirrored_strategy.scope():
                self.model.compile(optimizer=self.config.OPTIMIZER(learning_rate=self.config.LR),
                                   loss=[self.config.LOSS],
                                   metrics=[self.config.METRIC])
        else:
            self.model.compile(optimizer=self.config.OPTIMIZER(learning_rate=self.config.LR),
                               loss=[self.config.LOSS],
                               metrics=[self.config.METRIC])
        self.iscompiled = True

    def train(self,train=None,valid=None,callbacks=None):
        config = self.config

        if not self.iscompiled:
            self.compile()

        start = time.time()
        if valid == None:
            validation_split = config.VALIDATION_SPLIT if 'VALIDATION_SPLIT' in config.config.keys() else 0.1
            history = self.model.fit(train[0],train[1],
                                     batch_size=config.BATCH_SIZE,
                                     epochs=config.EPOCHS,
                                     verbose=config.FIT_VERBOSE,
                                     validation_split=validation_split,
                                     callbacks=callbacks)
        else:
            history = self.model.fit(train,
                                     batch_size=config.BATCH_SIZE,
                                     steps_per_epoch=len(train),
                                     epochs=config.EPOCHS,
                                     verbose=config.FIT_VERBOSE,
                                     validation_data=valid,
                                     validation_steps=len(valid),
                                     use_multiprocessing=True,
                                     workers=12,
                                     callbacks=callbacks)
        end = time.time()

        self.timeoftrain = end - start

        print('Time of training:', self.timeoftrain)
        self.history = history

    def evaluate(self,model_path,test=None):
        # assert test is not None
        config = self.config

        if not self.iscompiled:
            # self.model.compile(optimizer=config.OPTIMIZER(lr=config.LR), loss=[config.LOSS], metrics=[config.METRIC])
            self.compile()

        self.model.load_weights(model_path)
        print("Loaded: ",model_path)
        start = time.time()
        pred = self.model.predict(test, batch_size=config.BATCH_SIZE)
        end = time.time()
        self.timeofpred = end - start
        print('Time of prediction:', self.timeofpred)

        return pred
