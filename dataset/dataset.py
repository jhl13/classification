# encoding = utf-8
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from config.config import cfg
from utils import io_

class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.root_dir    = cfg.CLS.DATASET_ROOT_DIR
        self.file_path   = cfg.TRAIN.FILE_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.classes = io_.read_class_names(cfg.CLS.CLASSES )
        self.num_classes = len(self.classes)

        self.sample_paths = self.load_annotations()
        self.num_samples = len(self.sample_paths)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        print ("num_classes:%d, num_samples:%d, batch_size:%d, num_batchs:%d"\
            %(self.classes, self.num_samples, self.batch_size, self.num_batchs))

    def load_annotations(self):
        with open(self.file_path, 'r') as f:
            txt = f.readlines()
            annotations = [os.path.join(self.root_dir, line.strip()) for line in txt \
                if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations


