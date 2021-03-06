# encoding = utf-8
from easydict import EasyDict as edict

__C                             = edict()
# Consumers can get config by: from config import cfg
cfg                             = __C

# classification options
__C.CLS                         = edict()
__C.CLS.DATASET                 = "food-462" # ifood-251 or food-462
__C.CLS.CLASSES_462             = "./dataset/name.names"
__C.CLS.CLASSES_251             = "./dataset/class_list.txt"
__C.CLS.DATASET_ROOT_DIR        = "../datasets/baseline_food/"
__C.CLS.GPU                     = "2,3"

# train options
__C.TRAIN                       = edict()
__C.TRAIN.FILE_PATH_462         = "./dataset/train.txt"
__C.TRAIN.FILE_PATH_251         = "./dataset/train_labels.csv"
__C.TRAIN.RESNET_SIZE           = 50
__C.TRAIN.SAVE_DIR              = "../model_zoo/food_classification/food-462-sigmoid-focalloss"
__C.TRAIN.INPUT_SIZE            = 224
__C.TRAIN.BATCH_SIZE            = 32
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARNING_RATE         = 1e-3
__C.TRAIN.WEIGHT_DECAY          = 1e-4
__C.TRAIN.MOVING_AVE_DECAY      = 0.9
__C.TRAIN.INITIAL_WEIGHT        = "../model_zoo/food_classification/food-462-sigmoid-focalloss"
__C.TRAIN.TOTAL_EPOCHS          = 50

# test options
__C.TEST                        = edict()
__C.TEST.FILE_PATH_462          = "./dataset/test.txt"
__C.TEST.FILE_PATH_251          = "./dataset/val_labels.csv"
__C.TEST.INPUT_SIZE             = 224
__C.TEST.GPU                    = "1"
__C.TEST.BATCH_SIZE             = 32
__C.TEST.DATA_AUG               = False
__C.TEST.MODEL_ZOO              = "../model_zoo/food_classification/baseline"
__C.TEST.INITIAL_WEIGHT         = "../model_zoo/food_classification/baseline/resnet50_test_loss=2.8496.ckpt-15"
