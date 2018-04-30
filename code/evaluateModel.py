import tensorflow as tf
import numpy as np
import os
import datetime

from buildModel import CNN
from DataSetNPY import DataSetNPY
from metrics import *
from globalVars import *

class EvaluateModel:
    def __init__(self):
        dataBaseString = '/data/psturm/FETAL/'
        
        valdImagePrefix = 'valdImages/'
        valdLabelPrefix = 'valdMasks/mask_'
        
        testImagePrefix = 'testImages/'
        testLabelPrefix = 'testMasks/mask_'
        
        evalImageBatchDims = (-1, width, height, 37)
        
        valdDataSet = DataSetNPY(filenames=os.listdir('{}{}'.format(dataBaseString, valdImagePrefix)),
                                      imageBaseString='{}{}'.format(dataBaseString, valdImagePrefix),
                                      imageBatchDims=evalImageBatchDims,
                                      labelBatchDims=evalImageBatchDims,
                                      labelBaseString='{}{}'.format(dataBaseString, valdLabelPrefix))
        testDataSet = DataSetNPY(filenames=os.listdir('{}{}'.format(dataBaseString, testImagePrefix)),
                                      imageBaseString='{}{}'.format(dataBaseString, testImagePrefix),
                                      imageBatchDims=evalImageBatchDims,
                                      labelBatchDims=evalImageBatchDims,
                                      labelBaseString='{}{}'.format(dataBaseString, testLabelPrefix))
        valdDataSet.PreloadData()
        testDataSet.PreloadData()
        valdImages, valdMasks = valdDataSet.NextBatch(None)
        testImages, testMasks = testDataSet.NextBatch(None)
        
        valdImages = tf.reshape(
            tf.constant(valdImages, dtype=tf.float32, name='valdImages'),
            shape=(-1, width, height, 1)
        )
        valdMasks = tf.reshape(
            tf.constant(valdMasks, dtype=tf.float32, name='valdMasks'),
            shape=(-1, width, height)
        )
        testImages = tf.reshape(
            tf.constant(testImages, dtype=tf.float32, name='testImages'),
            shape=(-1, width, height, 1)
        )
        testMasks = tf.reshape(
            tf.constant(testMasks, dtype=tf.float32, name='testMasks'),
            shape=(-1, width, height)
        )
        
        with tf.device('/device:GPU:1'):
            self.useVald = tf.placeholder(dtype=tf.bool)
            imageInputOp = tf.cond(self.useVald,
                                   true_fn=lambda: valdImages,
                                   false_fn=lambda: testImages,
                                   name='inputEvalImages')
            maskInputOp  = tf.cond(self.useVald,
                                   true_fn=lambda: valdMasks,
                                   false_fn=lambda: testMasks,
                                   name='inputEvalMasks')
            binaryLabels = tf.cast(tf.round(maskInputOp / 255.0), tf.int32)
            tf.get_variable_scope().reuse_variables()
            
            #Build Model
            outputLayer = CNN(imageInputOp, reuse=True)
            binaryOutputLayer = tf.argmax(outputLayer, axis=3, output_type=tf.int32)
            
            self.lossOp = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=binaryLabels, logits=outputLayer))
            tf.summary.scalar('CrossEntropyVald', self.lossOp)

            #Define Dice Coefficient
            self.diceOp = Dice(binaryLabels, binaryOutputLayer)
            tf.summary.scalar('DiceVald', self.diceOp)
            
    def GetPerformance(self, sess, useVald=True):
        return sess.run([self.lossOp, self.diceOp], feed_dict={self.useVald: useVald})