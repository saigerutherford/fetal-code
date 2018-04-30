import tensorflow as tf
import numpy as np
import os
import datetime

from buildModel import CNN
#from evaluateModel import GetValidationPerformance, GetTestPerformance
from DataSetNPY import DataSetNPY
from metrics import *

with tf.device('/device:GPU:0'):
    # Global Values
    width = 96
    height = 96
    n_channels = 1
    n_classes = 2
    imageBatchDims = (-1, width, height, n_channels)
    labelBatchDims = (-1, width, height)

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
    imagesPL, labelsPL = trainDataSet.GetBatchOperations()

    # Define Placeholders
    binaryLabels = tf.cast(tf.round(labelsPL / 255.0), tf.int32)

    globalStep = tf.Variable(0, trainable=False)
    initLearningRate = 0.0001
    learningRate = tf.train.exponential_decay(initLearningRate, globalStep,
                                               1000, 0.9, staircase=True)
    tf.summary.scalar('learningRate', learningRate)

    #Build Model
    outputLayer = CNN(imagesPL)
    binaryOutputLayer = tf.argmax(outputLayer, axis=3, output_type=tf.int32)

    #Define Training Operation
    lossOp = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=binaryLabels, logits=outputLayer))
    trainOp = tf.train.AdamOptimizer(learningRate).minimize(lossOp, global_step=globalStep)
    tf.summary.scalar('CrossEntropyTrain', lossOp)

    #Define Dice Coefficient
    diceOp = Dice(binaryLabels, binaryOutputLayer)
    tf.summary.scalar('DiceTrain', diceOp)

    #Initialize weights, sessions
    timeString = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    modelPath   = '../checkpoints/{}'.format(timeString)
    summaryPath = '../summaries/{}'.format(timeString)
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    if not os.path.exists(summaryPath):
        os.makedirs(summaryPath)
    cappedIterations = 1000
    batchStepsBetweenSummaries = 10
    
    

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #Start Queue Runner for Data Input
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    #Define Summary Writer
    mergedSummaries = tf.summary.merge_all()
    summaryWriter = tf.summary.FileWriter(summaryPath, sess.graph)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for i in range(cappedIterations):        
        if i % batchStepsBetweenSummaries == 0:
            _, trainingLoss, diceCoeff, summaries = sess.run([trainOp, lossOp, diceOp, mergedSummaries])
            summaryWriter.add_summary(summaries, i)
            print('Iteration {}: Training Loss = {}, Dice = {}'.format(i, trainingLoss, diceCoeff))
        else:
            sess.run([trainOp])
            
    coord.request_stop()
    coord.join(threads)