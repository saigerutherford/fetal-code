import tensorflow as tf
import numpy as np
import os
import datetime

from medpy.io import load, save
from buildModel import CNN
from dataHelper import GetDataSets
from DataSetNPY import DataSetNPY
from metrics import *
from globalVars import *

def createMasks(fileNames, saveNames, checkpointDir=None):
    """
    Reloads a model from checkpointDir and runs it on the files
    specified in fileNames. For each file in fileNames,
    writes a mask whose filename is specified by saveNames.
    That is, the mask for fileNames[i] will be stored in saveNames[i].
    """
    if checkpointDir is None:
        checkpoints = os.listdir('../checkpoints')
        checkpointDir = '../checkpoints/' + np.sort(checkpoints)[-1] + '/'

    with tf.variable_scope('InputPlaceholders'):
        imageShape = [None, height, width, n_channels]
        imagesPL = tf.placeholder(dtype=tf.float32, shape=imageShape, name='imagesPL')

    #Build Model
    outputLayer = CNN(imagesPL)
    binaryOutputLayer = tf.argmax(outputLayer, axis=3, output_type=tf.int32)

    # Each individual medpy image is of shape (96, 96, 37)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, checkpointDir)
        print('Restored model from: {}'.format(checkpointDir))
        for i in range(len(fileNames)):
            fileName = fileNames[i]
            saveName = saveNames[i]

            image, header = load(fileName)
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, -1)

            #The predicted mask is of shape (37, 96, 96)
            predictedMask = sess.run(binaryOutputLayer, feed_dict={imagesPL: image})
            print(predictedMask)
            predictedMask = np.transpose(predictedMask, (1, 2, 0))
            save(predictedMask, saveName, header)

fileNames = ['zpr_2006-T1_run1_vol0017.nii']
saveNames = ['pred_zpr_2006-T1_run1_vol0017.nii']
createMasks(fileNames, saveNames)