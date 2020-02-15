# encoding = utf-8
from easydict import EasyDict as edict

__C                             = edict()
# Consumers can get config by: from config import cfg
cfg                             = __C

# classification options
__C.CLS                         = edict()
__C.CLS.CLASSES                 = "./dataset/name.names"
__C.CLS.DATASET_ROOT_DIR        = "/home/luo13/workspace/dataset/dish"

# train options
__C.TRAIN                       = edict()
__C.TRAIN.FILE_PATH             = "./dataset/train.txt"
__C.TRAIN.RESNET_SIZE           = 50
__C.TRAIN.SAVE_DIR              = "/home/luo13/workspace/model_zoo"
__C.TRAIN.INPUT_SIZE            = 224
__C.TRAIN.BATCH_SIZE            = 16
__C.TRAIN.DATA_AUG              = True

# test options
__C.TEST                        = edict()
__C.TEST.FILE_PATH              = "./dataset/test.txt"
__C.TEST.INPUT_SIZE             = 224
__C.TEST.BATCH_SIZE             = 1
__C.TEST.DATA_AUG               = False
