# encoding = utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf  # pylint: disable=g-bad-import-order

from models import resnet as resnet_model
from dataset.dataset import Dataset, preprocessing_train, preprocessing_test
from utils import gpu as gpu_util
from config.config import cfg
from tqdm import trange
from models.mobilenet import mobilenet_v2
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
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CLS.GPU

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

class CLSTrain(object):
    def __init__(self):
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.steps_per_period    = len(self.trainset)
        self.steps_test          = len(self.testset)
        self.moving_ave_decay    = cfg.TRAIN.MOVING_AVE_DECAY
        self.weight_decay        = cfg.TRAIN.WEIGHT_DECAY
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.total_epochs        = cfg.TRAIN.TOTAL_EPOCHS
        self.save_dir            = cfg.TRAIN.SAVE_DIR
        self.learning_rate       = cfg.TRAIN.LEARNING_RATE
        # self.mirrored_strategy = tf.distribute.MirroredStrategy()# 感觉只是做了分配数据这一步
        self.GPU_NUM             = len(cfg.CLS.GPU) if len(cfg.CLS.GPU) == 1 else len(cfg.CLS.GPU) - 1
        self.gpus                = gpu_util.get_available_gpus(self.GPU_NUM)
        self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.train_splited_batch_size = cfg.TRAIN.BATCH_SIZE // self.GPU_NUM
        self.test_splited_batch_size = cfg.TEST.BATCH_SIZE // self.GPU_NUM
        self.image_size = cfg.TRAIN.INPUT_SIZE
        self.clone_scopes = ['clone_%d'%(idx) for idx in range(len(self.gpus))]
        print ("steps_per_period:{}, steps_test:{}".format(self.steps_per_period, self.steps_test))
        if os.path.exists(self.save_dir) is not True:
            os.mkdir(self.save_dir)

        with tf.name_scope('define_input'):
            self.trainable    = tf.compat.v1.placeholder(dtype=tf.bool, name='training')
            with tf.device('/cpu:0'):
                train_dataset = tf.data.Dataset.from_generator(lambda: self.trainset, \
                    output_types=(tf.string, tf.float32), output_shapes=(tf.TensorShape([None]), tf.TensorShape([None, NUM_CLASSES])))
                train_dataset = train_dataset.prefetch(buffer_size=200)
                train_dataset = train_dataset.repeat()
                # train_dataset = train_dataset.map(lambda x, y: (x, y), num_parallel_calls=4)
                train_dataset = train_dataset.map(self.py_func_preprocess_train, num_parallel_calls=10)
                train_dataset_iter = train_dataset.make_one_shot_iterator()

                test_dataset = tf.data.Dataset.from_generator(lambda: self.testset, \
                    output_types=(tf.string, tf.float32), output_shapes=(tf.TensorShape([None]), tf.TensorShape([None, NUM_CLASSES])))
                test_dataset = test_dataset.prefetch(buffer_size=50)
                test_dataset = test_dataset.repeat()
                test_dataset = test_dataset.map(self.py_func_preprocess_test, num_parallel_calls=2)
                test_dataset_iter = test_dataset.make_one_shot_iterator()

                # NHWC
                batch_image, batch_label = \
                tf.cond(self.trainable, lambda: train_dataset_iter.get_next(), lambda: test_dataset_iter.get_next())

                splited_batch_size = \
                tf.cond(self.trainable, lambda: self.train_splited_batch_size, lambda: self.test_splited_batch_size)

        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            self.base = tf.Variable(0.1, dtype=tf.float32, trainable=False, name='base')
            self.global_step_update = tf.compat.v1.assign_add(self.global_step, 1.0)
            epoch_10 = tf.cast((self.global_step // self.steps_per_period) // 20, dtype=tf.float32)
            self.learning_rate = self.learning_rate * tf.pow(self.base, epoch_10)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        self.total_loss = 0
        # self.total_l2_loss = 0
        self.acc = 0
        total_clone_gradients = []
        for clone_idx, gpu in enumerate(self.gpus):
            reuse = clone_idx > 0
            with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse = reuse):
                with tf.name_scope(self.clone_scopes[clone_idx]) as clone_scope:
                    with tf.device(gpu) as clone_device:
                        tf.summary.image("batch_image", batch_image[clone_idx*splited_batch_size:(clone_idx+1)*splited_batch_size, :, :, :], 3)
                        resnet_model = CLSModel(resnet_size=50, data_format="channels_last") # CPU只支持channels_last
                        final_dense = resnet_model(batch_image[clone_idx*splited_batch_size:(clone_idx+1)*splited_batch_size, :, :, :], self.trainable)
                        # with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=self.trainable)):
                        #         final_dense, endpoints = \
                        #             mobilenet_v2.mobilenet(batch_image[clone_idx*splited_batch_size:(clone_idx+1)*splited_batch_size, :, :, :], \
                        #                 num_classes=NUM_CLASSES)

                        labels_per_gpu = batch_label[clone_idx*splited_batch_size:(clone_idx+1)*splited_batch_size, :]
                        # cross_entropy = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(logits=final_dense, labels=labels_per_gpu)
                        cross_entropy = self.focal_loss_sigmoid(final_dense, labels_per_gpu)
                        correct_prediction = tf.equal(tf.cast(tf.argmax(tf.nn.softmax(final_dense), 1), dtype=tf.int32), \
                            tf.cast(tf.argmax((labels_per_gpu), 1), dtype=tf.int32))
                        
                        # # Add weight decay to the loss.
                        # l2_loss = self.weight_decay * tf.add_n(
                        #     # loss is computed using fp32 for numerical stability.
                        #     [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
                        #     if self.exclude_batch_norm(v.name)])

                        self.acc += tf.reduce_mean(tf.cast(correct_prediction, "float"))
                        clone_loss = cross_entropy
                        self.total_loss += clone_loss
                        # self.total_l2_loss += l2_loss
                        clone_gradients = self.optimizer.compute_gradients(clone_loss, var_list=tf.trainable_variables())
                        total_clone_gradients.append(clone_gradients)
        average_gradients = self.sum_gradients(total_clone_gradients)
        grad_op = self.optimizer.apply_gradients(average_gradients)

        # shadow_variable = decay * shadow_variable + (1 - decay) * variable
        with tf.name_scope("define_weight_decay"):
            self.moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())
        
        # 会先执行定义的操作，再执行后续的操作
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([grad_op, self.global_step_update]):
                with tf.control_dependencies([self.moving_ave]):
                    self.train_op = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(tf.global_variables())
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=3)

        self.total_loss = self.total_loss / self.GPU_NUM
        self.acc = self.acc / self.GPU_NUM
        # self.total_l2_loss = self.total_l2_loss / self.GPU_NUM

        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar("acc", self.acc)

        self.write_op = tf.summary.merge_all()
        self.summary_writer  = tf.summary.FileWriter(self.save_dir, graph=self.sess.graph)

    def sum_gradients(self, clone_grads):
        """计算梯度
        Arguments:
            clone_grads -- 每个GPU所对应的梯度
        Returns:
            averaged_grads -- 平均梯度
        """                  
        averaged_grads = []
        for grad_and_vars in zip(*clone_grads):
            grads = []
            var = grad_and_vars[0][1]
            try:
                for g, v in grad_and_vars:
                    assert v == var
                    grads.append(g)
                grad = tf.add_n(grads, name = v.op.name + '_summed_gradients')
            except:
                import pdb
                pdb.set_trace()
            averaged_grads.append((grad, v))
        return averaged_grads

    def focal_loss_softmax(self, input_data, one_hot_label):
        softmax_input = tf.nn.softmax(input_data)
        p_true = tf.reduce_sum(one_hot_label*softmax_input, 1)
        focal_loss = tf.reduce_sum(-tf.pow((1 - p_true), 2) * tf.reduce_sum(one_hot_label*tf.log(softmax_input), 1), 0) # 用该批次里最少的同一类样本数进行归一化
        return focal_loss

    def focal_loss_sigmoid(self, input_data, one_hot_label): # 可能存在多个标签
        sigmoid_input = tf.nn.sigmoid(input_data)
        focal_loss_pos = tf.reduce_sum(\
            -tf.reduce_sum(tf.pow(one_hot_label * (1 - sigmoid_input), 2), 1) * \
                tf.reduce_sum(one_hot_label * tf.log(sigmoid_input), 1), 0) # 保证标签为1的预测值都是1
        
        focal_loss_neg = tf.reduce_sum(\
            -tf.reduce_sum(tf.pow((1 - one_hot_label) * (sigmoid_input), 2), 1) * \
                tf.reduce_sum((1 - one_hot_label) * tf.log(1 - sigmoid_input), 1), 0) # 保证标签为0的预测值都是0

        focal_loss = (focal_loss_pos + focal_loss_neg)
        return focal_loss

    def exclude_batch_norm(self, name):
        return 'batch_normalization' not in name

    def py_func_preprocess_train(self, tensor_image_paths, tensor_label):
        tensor_image, tensor_label_change = tf.compat.v1.py_func(preprocessing_train, [tensor_image_paths, tensor_label], [tf.float32, tf.float32])
        tensor_image_reshape = tf.cast(tf.reshape(tensor_image, [cfg.TRAIN.BATCH_SIZE, self.image_size, self.image_size, 3]), dtype=tf.float32)
        tensor_label = tf.cast(tensor_label, dtype=tf.float32)
        return tensor_image_reshape, tensor_label_change

    def py_func_preprocess_test(self, tensor_image_paths, tensor_label):
        tensor_image, tensor_label_change = tf.compat.v1.py_func(preprocessing_test, [tensor_image_paths, tensor_label], [tf.float32, tf.float32])
        tensor_image_reshape = tf.cast(tf.reshape(tensor_image, [cfg.TEST.BATCH_SIZE, self.image_size, self.image_size, 3]), dtype=tf.float32)
        tensor_label = tf.cast(tensor_label, dtype=tf.float32)
        return tensor_image_reshape, tensor_label_change

    def train(self):
        test_best_loss = 0
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, tf.train.latest_checkpoint(self.initial_weight))
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            ckpt_file = os.path.join(self.save_dir, "initial.ckpt")
            self.saver.save(self.sess, ckpt_file, global_step=self.global_step)

        for epoch in range(1, 1+self.total_epochs):
            pbar = trange(self.steps_per_period)
            test = trange(self.steps_test)
            train_epoch_loss = []
            test_epoch_loss = []
            test_epoch_acc = []
            for i in pbar:
                _, train_step_loss, summary, global_step_value = self.sess.run(
                    [self.train_op, self.total_loss, self.write_op, self.global_step],feed_dict={self.trainable: True})
                train_epoch_loss.append(train_step_loss)
                pbar.set_description("train loss: %.2f" %(train_step_loss))
                if int(global_step_value) % 100 == 0:
                    self.summary_writer.add_summary(summary, global_step_value)
        
            for _ in test:
                test_step_loss, test_step_acc = self.sess.run([self.total_loss, self.acc], feed_dict={self.trainable: False})
                test_epoch_loss.append(test_step_loss)
                test_epoch_acc.append(test_step_acc)
                test.set_description("test loss: %.2f" %(test_step_loss))

            train_epoch_loss, test_epoch_loss, test_epoch_acc = np.mean(train_epoch_loss), np.mean(test_epoch_loss), np.mean(test_epoch_acc)
            ckpt_file = os.path.join(self.save_dir, "resnet50_test_loss=%.4f_acc=%.2f.ckpt" % (test_epoch_loss, test_epoch_acc))
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            if epoch == 1:
                test_best_loss = test_epoch_loss
            if test_epoch_loss <= test_best_loss:
                self.saver.save(self.sess, ckpt_file, global_step=epoch)
                print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
                test_best_loss = test_epoch_loss
            else:
                print("=> Epoch: %2d Time: %s we don't save model this epoch ...\n loss=%.4f acc=%.2f"
                                %(epoch, log_time, test_epoch_loss, test_epoch_acc))

if __name__ == '__main__': CLSTrain().train()
