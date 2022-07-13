from importlib import import_module
import time


class BaseModel:
    '''
    BaseModel class used in initial work. It was designed to use keras functional api.
    To use class api please create its realization.
    '''
    def __init__(self, config):
        self.config = config

        #Generate the model
        try:
            model_name = config.MODEL_NAME
            print('Importing {0:s}'.format(model_name))
            module = import_module('models.' + model_name)
            model_gen = module.model_gen
        except:
            print('Error: module {0:s} is not exist!'.format(model_name))

        self.iscompiled = False
        self.model = model_gen(config)

    def summary(self):
        self.model.summary(line_length=120)

    def compile(self):
        self.model.compile(optimizer=self.config.OPTIMIZER(lr=self.config.LR),
                           loss=[self.config.LOSS],
                           metrics=[self.config.METRIC])
        self.iscompiled = True

    def train(self,train=None,valid=None,callbacks=None):
        config = self.config

        if not self.iscompiled:
            self.compile()

        start = time.time()
        if valid == None:
            validation_split = config.VALIDATION_SPLIT if 'VALIDATION_SPLIT' is in config.keys else 0.1
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
            self.model.compile(optimizer=config.OPTIMIZER(lr=config.LR), loss=[config.LOSS], metrics=[config.METRIC])
            self.iscompiled = True

        self.model.load_weights(model_path)
        print("Loaded: ",model_path)
        start = time.time()
        pred = self.model.predict(test, batch_size=config.BATCH_SIZE)
        end = time.time()
        self.timeofpred = end - start
        print('Time of prediction:', self.timeofpred)

        return pred
