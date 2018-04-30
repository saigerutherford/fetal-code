import time
import os

from medpy.io import load, save
import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d

import tensorflow as tf

def CNN(input, reuse=False):
    width = 96
    height = 96 
    n_channels = 1
    n_classes = 2
    ################Create Model######################
    conv1 = conv_2d(input, 32, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv1')
    conv1 = conv_2d(conv1, 32, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv2')
    pool1 = max_pool_2d(conv1, 2)

    conv2 = conv_2d(pool1, 64, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv3')
    conv2 = conv_2d(conv2, 64, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv4')
    pool2 = max_pool_2d(conv2, 2)

    conv3 = conv_2d(pool2, 128, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv5')
    conv3 = conv_2d(conv3, 128, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv6')
    pool3 = max_pool_2d(conv3, 2)

    conv4 = conv_2d(pool3, 256, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv7')
    conv4 = conv_2d(conv4, 256, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv8')
    pool4 = max_pool_2d(conv4, 2)

    conv5 = conv_2d(pool4, 512, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv9')
    conv5 = conv_2d(conv5, 512, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv10')

    up6 = upsample_2d(conv5,2)
    up6 = tflearn.layers.merge_ops.merge([up6, conv4], 'concat',axis=3)
    conv6 = conv_2d(up6, 256, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv11')
    conv6 = conv_2d(conv6, 256, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv12')

    up7 = upsample_2d(conv6,2)
    up7 = tflearn.layers.merge_ops.merge([up7, conv3],'concat', axis=3)
    conv7 = conv_2d(up7, 128, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv13')
    conv7 = conv_2d(conv7, 128, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv14')

    up8 = upsample_2d(conv7,2)
    up8 = tflearn.layers.merge_ops.merge([up8, conv2],'concat', axis=3)
    conv8 = conv_2d(up8, 64, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv15')
    conv8 = conv_2d(conv8, 64, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv16')

    up9 = upsample_2d(conv8,2)
    up9 = tflearn.layers.merge_ops.merge([up9, conv1],'concat', axis=3)
    conv9 = conv_2d(up9, 32, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv17')
    conv9 = conv_2d(conv9, 32, 3, activation='relu', padding='same', regularizer="L2", reuse=reuse, scope='conv18')

    pred = conv_2d(conv9, 2, 1,  activation='linear', padding='valid', reuse=reuse, scope='conv19')

    pred_reshape = tf.reshape(pred, [-1, width, height, n_classes])
    return pred_reshape
