import pandas as pd
import cv2
import os
import numpy as np
from utils import str_
import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
a = tf.placeholder(dtype=tf.float32, shape=(2,3))
a_label = tf.placeholder(dtype=tf.float32, shape=(2,3))

sigmoid_input = tf.nn.sigmoid(a)
focal_loss_pos = tf.reduce_mean(\
    -tf.reduce_sum(tf.pow(a_label * (1 - sigmoid_input), 2), 1) * \
        tf.reduce_sum(a_label * tf.log(sigmoid_input), 1), 0) # 保证标签为1的预测值都是1

c = -tf.reduce_sum(tf.pow(a_label * (1 - sigmoid_input), 2), 1) * \
        tf.reduce_sum(a_label * tf.log(sigmoid_input), 1)

focal_loss_neg = tf.reduce_mean(\
    -tf.reduce_sum(tf.pow((1 - a_label) * (sigmoid_input), 2), 1) * \
        tf.reduce_sum((1 - a_label) * tf.log(1 - sigmoid_input), 1), 0) # 保证标签为0的预测值都是0

d = -tf.reduce_sum(tf.pow((1 - a_label) * (sigmoid_input), 2), 1) * \
        tf.reduce_sum((1 - a_label) * tf.log(1 - sigmoid_input), 1)


focal_loss = focal_loss_pos + focal_loss_neg

softmax_input = tf.nn.softmax(a)
p_true = tf.reduce_sum(a_label*softmax_input, 1)
focal_loss = tf.reduce_sum(-tf.pow((1 - p_true), 2) * tf.reduce_sum(a_label*tf.log(softmax_input), 1), 0) \
    / tf.reduce_max(tf.reduce_sum(a_label, 1)) # 用该批次里最少的同一类样本数进行归一化

a_numpy = np.asarray([[-10, -10, 10],[10, -10, -10]], dtype=np.float32)
a_label_numpy = np.asarray([[0, 0, 1],[1, 0, 0]], dtype=np.float32)

c_np, d_np, p_np = sess.run([focal_loss_pos, focal_loss_neg, focal_loss], feed_dict={a:a_numpy, a_label:a_label_numpy})
print (c_np, d_np, p_np)
