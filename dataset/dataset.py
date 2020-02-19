# encoding = utf-8
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from config.config import cfg
from utils import io_, str_
from pandas import read_csv

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]

class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG
        self.root_dir    = cfg.CLS.DATASET_ROOT_DIR
        assert cfg.CLS.DATASET in ["ifood-251", "food-462"]
        if cfg.CLS.DATASET == "food-462":
            self.file_path   = cfg.TRAIN.FILE_PATH_462 if dataset_type == 'train' else cfg.TEST.FILE_PATH_462
            self.classes = io_.read_class_names_462(cfg.CLS.CLASSES_462)
            self.sample_paths = self.load_annotations_462()
            self.num_classes = len(self.classes)
            self.num_samples = len(self.sample_paths)
        else:
            self.file_path   = cfg.TRAIN.FILE_PATH_251 if dataset_type == 'train' else cfg.TEST.FILE_PATH_251
            self.classes = io_.read_class_names_251(cfg.CLS.CLASSES_251)
            self.sample_paths = self.load_annotations_251()
            self.num_classes = len(self.classes)
            self.num_samples = len(self.sample_paths)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        print ("num_classes:%d, num_samples:%d, batch_size:%d, num_batchs:%d"\
            %(self.num_classes, self.num_samples, self.batch_size, self.num_batchs))

    def load_annotations_462(self):
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

    def load_annotations_251(self):
        ann = read_csv(self.file_path)
        labels_paths = []
        if self.dataset_type == "train":
            train_dir = "ifood-2019-fgvc6/train_set"
            ann['path'] = ann['img_name'].map(lambda x: os.path.join(self.root_dir, os.path.join(train_dir, x)))
        else:
            val_dir = "ifood-2019-fgvc6/val_set"
            ann['path'] = ann['img_name'].map(lambda x: os.path.join(self.root_dir, os.path.join(val_dir, x)))
        for i in range(len(ann)):
            label_path = [ann['label'][i], ann['path'][i]]
            labels_paths.append(label_path)
        np.random.shuffle(labels_paths)
        return labels_paths

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            # batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))
            batch_path = []
            # batch_label = np.zeros((self.batch_size, self.num_classes))
            batch_label = np.zeros((self.batch_size))
            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    sample_path = self.sample_paths[index]
                    # label, image = self.parse_sample(sample_path)
                    label, image_path = self.parse_sample(sample_path)
                    # batch_image[num, :, :, :] = image
                    batch_path.append(image_path)
                    batch_label[num] = label
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
        if isinstance(label_name, str) is True:
            label = self.classes.get(label_name, 0)
        else:
            label = label_name
        # onehot = np.zeros(self.num_classes, dtype=np.float)
        # onehot[label] = 1.0
        # return onehot, image_resized
        return label, image_path
    
    def __len__(self):
        return self.num_batchs

def preprocessing_train(image_paths, label):
    batch_image = np.zeros((len(image_paths), cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3), dtype=np.float32)
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        image = np.array(cv2.imread(image_path.decode()), dtype=np.float32)
        image = crop_and_flip(image)
        image_resized = cv2.resize(image, (cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE))
        image_resized = image_resized - _CHANNEL_MEANS
        batch_image[i, :, :, :] = image_resized
    return batch_image, label

def preprocessing_test(image_paths, label):
    batch_image = np.zeros((len(image_paths), cfg.TEST.INPUT_SIZE, cfg.TEST.INPUT_SIZE, 3), dtype=np.float32)
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        image = np.array(cv2.imread(image_path.decode()), dtype=np.float32)
        image_resized = cv2.resize(image, (cfg.TEST.INPUT_SIZE, cfg.TEST.INPUT_SIZE))
        image_resized = image_resized - _CHANNEL_MEANS
        batch_image[i, :, :, :] = image_resized
    return batch_image, label

def crop_and_flip(image):
    if random.random() < 0.5: # horizontal
        image = image[:, ::-1, :]

    if random.random() < 0.5: # vertical
        image = image[::-1, :, :]

    if random.random() < 0.5: # crop from central
        h, w, _ = image.shape
        max_l = w / 2 - cfg.TRAIN.INPUT_SIZE / 2
        max_u = h / 2 - cfg.TRAIN.INPUT_SIZE / 2
        max_r = w / 2 + cfg.TRAIN.INPUT_SIZE / 2
        max_d = h / 2 + cfg.TRAIN.INPUT_SIZE / 2
        crop_xmin = max(0, int(max_l - random.uniform(0, max_l)))
        crop_ymin = max(0, int(max_u - random.uniform(0, max_u)))
        crop_xmax = min(w, int(max_r + random.uniform(0, w - max_r)))
        crop_ymax = min(h, int(max_d + random.uniform(0, h - max_d)))
        image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
    return image

def random_translate(image):
    if random.random() < 0.5:
        h, w, _ = image.shape

        max_l = w / 2 - cfg.TRAIN.INPUT_SIZE / 2
        max_u = h / 2 - cfg.TRAIN.INPUT_SIZE / 2
        max_r = w / 2 + cfg.TRAIN.INPUT_SIZE / 2
        max_d = h / 2 + cfg.TRAIN.INPUT_SIZE / 2

        tx = random.uniform(-(max_u - 1), (max_r - 1))
        ty = random.uniform(-(max_u - 1), (max_d - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))
    return image
