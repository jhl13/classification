# encoding = utf-8
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from config.config import cfg
from utils import io_, str_

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.root_dir    = cfg.CLS.DATASET_ROOT_DIR
        self.file_path   = cfg.TRAIN.FILE_PATH if dataset_type == 'train' else cfg.TEST.FILE_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_size = cfg.TRAIN.INPUT_SIZE # 改成多尺度？
        self.classes = io_.read_class_names(cfg.CLS.CLASSES )
        self.num_classes = len(self.classes)

        self.sample_paths = self.load_annotations()
        self.num_samples = len(self.sample_paths)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        print ("num_classes:%d, num_samples:%d, batch_size:%d, num_batchs:%d"\
            %(self.num_classes, self.num_samples, self.batch_size, self.num_batchs))

    def load_annotations(self):
        with open(self.file_path, 'r') as f:
            labels_paths = []
            txt = f.readlines()
            for line in txt:
                if len(line) != 0:
                    line = str_.remove_all(line, '\xef\xbb\xbf')
                    line = str_.remove_all(line, '\n')
                    label_name, path = str_.split(line, ',')
                    path = os.path.join(self.root_dir, path)
                    label_path = [label_name, path]
                    labels_paths.append(label_path)
        np.random.shuffle(labels_paths)
        return labels_paths

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            # batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))
            batch_path = []
            batch_label = np.zeros((self.batch_size, self.num_classes))
            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        print (index) 
                        index -= self.num_samples
                    sample_path = self.sample_paths[index]
                    # label, image = self.parse_sample(sample_path)
                    label, image_path = self.parse_sample(sample_path)
                    # batch_image[num, :, :, :] = image
                    batch_path.append(image_path)
                    batch_label[num, :] = label
                    num += 1
                self.batch_count += 1
                # return batch_image, batch_label
                return batch_path, batch_label
            else:
                self.batch_count = 0
                np.random.shuffle(self.sample_paths)
                raise StopIteration

    def parse_sample(self, sample_path):
        label_name, image_path = sample_path
        # image = np.array(cv2.imread(image_path))
        # image_resized = cv2.resize(image, (self.input_sizes, self.input_sizes))
        label = self.classes.get(label_name, 0)
        onehot = np.zeros(self.num_classes, dtype=np.float)
        onehot[label] = 1.0
        # return onehot, image_resized
        return onehot, image_path
    
    def __len__(self):
        return self.num_batchs

def preprocessing_train(image):
    pass

def preprocessing_test(image):
    pass

def crop_and_flip(image):
    
