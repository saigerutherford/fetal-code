import tensorflow as tf
import numpy as np
import os
import datetime

from buildModel import CNN
from DataSetNPY import DataSetNPY
from metrics import *
from globalVars import *


def GetDataSets():
    with tf.variable_scope('EvaluationInputs'):
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
    
    with tf.variable_scope('TrainingInputs'):
        #Define dataset operations
        dataBaseString = '/data/psturm/FETAL/'
        trainPrefix = 'trainImages/'
        labelPrefix = 'trainMasks/mask_'
        filenames = os.listdir('{}{}'.format(dataBaseString, trainPrefix))
        trainDataSet = DataSetNPY(filenames=filenames,
                                  imageBaseString='{}{}'.format(dataBaseString, trainPrefix),
                                  imageBatchDims=imageBatchDims,
                                  labelBatchDims=labelBatchDims,
                                  labelBaseString='{}{}'.format(dataBaseString, labelPrefix),
                                  batchSize=64)
    return trainDataSet, valdDataSet, testDataSet