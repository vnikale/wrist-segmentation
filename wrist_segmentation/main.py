import wrist_segmentation.data.preprocess as preprocess
import wrist_segmentation.data.loader as loader
from wrist_segmentation.utils.config import config
from wrist_segmentation.models import BaseModel


def run_train():
    conf = config('unet_al.yaml',log_config=1)

    folder = r'D:/new_SRW/net/dataset_new/whole_separated/*/*.mat'
    pre_args = {'preprocess_func':preprocess.full_preprocess,
               'size':conf.IMAGE_SIZE}
    args = {'folder':folder,
           'pre_args':pre_args,
            'limit':2,
           'config':conf}

    loader_ = loader.TrainTestDataloaderMat(**args)

    X, y, subj_len = loader_.load()


    model = BaseModel.BaseModel(conf)

    model.train(train=(X,y))

if __name__ == '__main__':
    run_train()