import tensorflow as tf
import numpy as np
import os
import datetime

from buildModel import CNN
from dataHelper import GetDataSets
from DataSetNPY import DataSetNPY
from metrics import *
from globalVars import *

def getPerformanceOnImages(sess, images, masks, imagesPL, labelsPL, lossOp, diceOp):
    accumulatedLoss = 0
    accumulatedDice = 0
    denom = 0
    for i in range(images.shape[-1]):
        patientImages = np.reshape(images[:, :, :, i], (-1, 96, 96, 1))
        patientMasks = np.reshape(masks[:, :, :, i], (-1, 96, 96))
        feed_dict = {
            imagesPL: patientImages,
            labelsPL: patientMasks
        }
        loss, dice = sess.run([lossOp, diceOp], feed_dict=feed_dict)
        accumulatedLoss += loss
        accumulatedDice += dice
        denom += 1
    return accumulatedLoss / denom, accumulatedDice / denom
    
with tf.device('/device:GPU:0'):
    #Define Dataset Operations
    trainDataSet, valdDataSet, testDataSet = GetDataSets()
    imageShape = [None, height, width, n_channels]
    maskShape = [None, height, width]
    with tf.variable_scope('InputPlaceholders'):
        imagesPL = tf.placeholder(dtype=tf.float32, shape=imageShape, name='imagesPL')
        labelsPL = tf.placeholder(dtype=tf.float32, shape=maskShape, name='masksPL')
        binaryLabels = tf.cast(labelsPL > 0, tf.int32)
    valdImages, valdMasks = valdDataSet.NextBatch(None) # Of shape (-1, 96, 96, 37)
    testImages, testMasks = testDataSet.NextBatch(None)
    
    imageShape = [-1, height, width, n_channels]
    maskShape = [-1, height, width]

    # Define Learning Parameters
    globalStep = tf.Variable(0, trainable=False)
    initLearningRate = 0.0001
    learningRate = tf.train.exponential_decay(initLearningRate, globalStep,
                                               1000, 0.9, staircase=True)
    tf.summary.scalar('learningRate', learningRate)

    #Build Model
    outputLayer = CNN(imagesPL)
    binaryOutputLayer = tf.argmax(outputLayer, axis=3, output_type=tf.int32)

    #Define Training Operation
    with tf.variable_scope('LossOperations'):
        lossOp = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=binaryLabels, logits=outputLayer))
        trainOp = tf.train.AdamOptimizer(learningRate).minimize(lossOp, global_step=globalStep)
        lossPL = tf.placeholder(dtype=tf.float32, shape=())
        tf.summary.scalar('CrossEntropy', lossPL)

        #Define Dice Coefficient
        diceOp = Dice(binaryLabels, binaryOutputLayer)
        dicePL = tf.placeholder(dtype=tf.float32, shape=())
        tf.summary.scalar('DiceCoeff', dicePL)

    #Initialize weights, sessions
    timeString = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    modelPath   = '../checkpoints/{}'.format(timeString)
    summaryPath = '../summaries/{}'.format(timeString)
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    if not os.path.exists(summaryPath):
        os.makedirs(summaryPath)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #Start Queue Runner for Data Input
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    #Define Summary Writer
    mergedSummaries = tf.summary.merge_all()
    trainWriter = tf.summary.FileWriter(summaryPath + '/train', sess.graph)
    evalWriter = tf.summary.FileWriter(summaryPath + '/eval')
    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    for i in range(cappedIterations):
        trainImages, trainMasks = trainDataSet.NextBatch(sess)
        feed_dict = {
            imagesPL: trainImages,
            labelsPL: trainMasks
        }
        
        if i % batchStepsBetweenSummaries == 0:
            _, trainingLoss, diceCoeff = sess.run([trainOp, lossOp, diceOp], feed_dict=feed_dict)
            summaries = sess.run(mergedSummaries, feed_dict={lossPL: trainingLoss, dicePL: diceCoeff})
            trainWriter.add_summary(summaries, i)
            
            valdLoss, valdDice = getPerformanceOnImages(sess, valdImages, valdMasks, imagesPL, labelsPL, lossOp, diceOp)
            summaries = sess.run(mergedSummaries, feed_dict={lossPL: valdLoss, dicePL: valdDice})
            evalWriter.add_summary(summaries, i)
            print('Iteration {}, Training: [Loss = {}, Dice = {}], Validation: [Loss = {}, Dice = {}]'.format(i, trainingLoss, diceCoeff, valdLoss, valdDice))
        else:
            sess.run([trainOp], feed_dict=feed_dict)
    
    testLoss, testDice = getPerformanceOnImages(sess, testImages, testMasks, imagesPL, labelsPL, lossOp, diceOp)
    print('Test: [Loss = {}, Dice = {}]'.format(testLoss, testDice))
    coord.request_stop()
    coord.join(threads)