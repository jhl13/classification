# encoding = utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf  # pylint: disable=g-bad-import-order

from models import resnet as resnet_model
from dataset.dataset import Dataset
from utils import gpu as gpu_util
from config.config import cfg
from tqdm import trange
import numpy as np
import time
import cv2

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 462

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

DATASET_NAME = 'ImageNet'
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TEST.GPU

def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)

class CLSModel(resnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
    else:
      bottleneck = True

    super(CLSModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )

def _preprocess_data(image_paths, label):
    batch_image = np.zeros((len(image_paths), 224, 224, 3), dtype=np.float32)
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        image = np.array(cv2.imread(image_path.decode()))
        image_resized = cv2.resize(image, (224, 224))
        batch_image[i, :, :, :] = image_resized
    return batch_image, label

def py_func_preprocess_data(tensor_image_paths, tensor_label):
    tensor_image, tensor_label_change = tf.compat.v1.py_func(_preprocess_data, [tensor_image_paths, tensor_label], [tf.float32, tf.float32])
    tensor_image_reshape = tf.cast(tf.reshape(tensor_image, [1, 224, 224, 3]), dtype=tf.float32)
    return tensor_image_reshape, tensor_label_change

class CLSTEST(object):
    def __init__(self):
      self.testset             = Dataset('test')
      self.steps_test          = len(self.testset)
      self.moving_ave_decay    = cfg.TRAIN.MOVING_AVE_DECAY
      self.initial_weight      = cfg.TEST.INITIAL_WEIGHT
      self.save_dir            = cfg.TEST.MODEL_ZOO
      # self.mirrored_strategy = tf.distribute.MirroredStrategy()# 感觉只是做了分配数据这一步
      self.GPU_NUM             = len(cfg.CLS.GPU) if len(cfg.CLS.GPU) == 1 else len(cfg.CLS.GPU) - 1
      self.gpus                = gpu_util.get_available_gpus(self.GPU_NUM)
      self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
      self.test_splited_batch_size = cfg.TEST.BATCH_SIZE
      self.image_size = cfg.TRAIN.INPUT_SIZE
      self.clone_scopes = ['clone_%d'%(idx) for idx in range(len(self.gpus))]
      print ("steps_test:{}".format(self.steps_test))
      assert os.path.exists(self.save_dir) is True

      self.trainable    = tf.compat.v1.placeholder(dtype=tf.bool, name='training')

      test_dataset = tf.data.Dataset.from_generator(lambda: self.testset, \
                  output_types=(tf.string, tf.float32), output_shapes=(tf.TensorShape([None]), tf.TensorShape([None, NUM_CLASSES])))
      test_dataset = test_dataset.repeat()
      test_dataset = test_dataset.map(py_func_preprocess_data, num_parallel_calls=2)
      test_dataset = test_dataset.prefetch(buffer_size=20)
      test_dataset_iter = test_dataset.make_one_shot_iterator()
      batch_image, self.batch_label = test_dataset_iter.get_next()
      splited_batch_size = self.test_splited_batch_size

      resnet_model = CLSModel(resnet_size=50, data_format="channels_last")
      final_dense = resnet_model(batch_image[:, :, :, :], self.trainable)
      self.p = tf.nn.softmax(final_dense)

      with tf.name_scope('emdefine_weight_decaya'):
          ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay) 

      self.saver = tf.train.Saver(ema_obj.variables_to_restore())
      self.saver.restore(self.sess, cfg.TEST.INITIAL_WEIGHT)

    def evaluate(self):
      test = trange(self.steps_test)
      for _ in test:
        p, label = self.sess.run([self.p, self.batch_label], feed_dict={self.trainable: False})
        print ("*"*10)
        print (np.max(p[0]))
        print (np.argmax(p[0]))
        print (np.argmax(label[0]))
        print (label[0])

if __name__ == '__main__': CLSTEST().evaluate()
