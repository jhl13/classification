# encoding = utf-8
from easydict import EasyDict as edict

__C                             = edict()
# Consumers can get config by: from config import cfg
cfg                             = __C

# classification options
__C.CLS                         = edict()
__C.CLS.CLASSES                 = "./dataset/name.names"
__C.CLS.DATASET_ROOT_DIR        = "../datasets/baseline_food/"
__C.CLS.GPU                     = "2,3"

# train options
__C.TRAIN                       = edict()
__C.TRAIN.FILE_PATH             = "./dataset/train.txt"
__C.TRAIN.RESNET_SIZE           = 50
__C.TRAIN.SAVE_DIR              = "../models/food_classification/baseline"
__C.TRAIN.INPUT_SIZE            = 224
__C.TRAIN.BATCH_SIZE            = 16
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARNING_RATE         = 1e-3
__C.TRAIN.MOVING_AVE_DECAY      = 0.9
__C.TRAIN.INITIAL_WEIGHT        = "../models/food_classification/baseline"
__C.TRAIN.TOTAL_EPOCHS          = 50

# test options
__C.TEST                        = edict()
__C.TEST.FILE_PATH              = "./dataset/test.txt"
__C.TEST.INPUT_SIZE             = 224
__C.TEST.BATCH_SIZE             = 1
__C.TEST.DATA_AUG               = False
