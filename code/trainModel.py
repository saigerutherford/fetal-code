import tensorflow as tf
import numpy as np
import os
import datetime

from buildModel import CNN
from dataHelper import GetDataSets
from DataSetNPY import DataSetNPY
from metrics import *
from globalVars import *

def getPerformanceOnImages(sess, images, masks, imagesPL, labelsPL, lossOp, diceOp, inputSummary):
    accumulatedLoss = 0
    accumulatedDice = 0
    denom = 0
    summaryInt = np.random.randint(0, images.shape[-1])
    valdInputSummary = None
    
    for i in range(images.shape[-1]):
        patientImages = np.reshape(images[:, :, :, i], (37, 96, 96, 1))
        patientMasks = np.reshape(masks[:, :, :, i], (37, 96, 96))
        feed_dict = {
            imagesPL: patientImages,
            labelsPL: patientMasks
        }
        if summaryInt == i and inputSummary is not None:
            loss, dice, valdInputSummary = sess.run([lossOp, diceOp, inputSummary], feed_dict=feed_dict)
        else:
            loss, dice = sess.run([lossOp, diceOp], feed_dict=feed_dict)
        accumulatedLoss += loss
        accumulatedDice += dice
        denom += 1
    return accumulatedLoss / denom, accumulatedDice / denom, valdInputSummary
    
with tf.device('/device:GPU:0'):
    #Define Dataset Operations
    trainDataSet, valdDataSet, testDataSet = GetDataSets()
    imageShape = [None, height, width, n_channels]
    maskShape = [None, height, width]
    with tf.variable_scope('InputPlaceholders'):
        imagesPL = tf.placeholder(dtype=tf.float32, shape=imageShape, name='imagesPL')
        labelsPL = tf.placeholder(dtype=tf.float32, shape=maskShape, name='masksPL')
        
        binaryLabels = tf.cast(labelsPL > 0, tf.int32)
    valdImages, valdMasks = np.swapaxes(valdDataSet.NextBatch(None), 0, 3) # Of shape (37, 96, 96, -1)
    testImages, testMasks = np.swapaxes(testDataSet.NextBatch(None), 0, 3)

    # Define Learning Parameters
    globalStep = tf.Variable(0, trainable=False)
    initLearningRate = 0.0001
    learningRate = tf.train.exponential_decay(initLearningRate, globalStep,
                                               10000, 0.9, staircase=True)
    learningRateSummary = tf.summary.scalar('learningRate', learningRate)

    #Build Model
    outputLayer = CNN(imagesPL)
    binaryOutputLayer = tf.argmax(outputLayer, axis=3, output_type=tf.int32)
    inputSummary = tf.summary.merge([tf.summary.image('Images', imagesPL), 
                                     tf.summary.image('TrueMasks', tf.expand_dims(binaryLabels, -1)),
                                     tf.summary.image('PredictedMasks', tf.expand_dims(binaryOutputLayer, -1))
                                    ])

    #Define Training Operation
    with tf.variable_scope('LossOperations'):
        lossOp = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=binaryLabels, logits=outputLayer))
        trainOp = tf.train.AdamOptimizer(learningRate).minimize(lossOp, global_step=globalStep)
        lossPL = tf.placeholder(dtype=tf.float32, shape=())
        entropySummary = tf.summary.scalar('CrossEntropy', lossPL)

        #Define Dice Coefficient
        diceOp = Dice(binaryLabels, binaryOutputLayer)
        evalDiceOp = PatientDice(binaryLabels, binaryOutputLayer)
        dicePL = tf.placeholder(dtype=tf.float32, shape=())
        diceSummary = tf.summary.scalar('DiceCoeff', dicePL)

    #Initialize weights, sessions
    timeString = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    modelPath   = '../checkpoints/{}/'.format(timeString)
    summaryPath = '../summaries/{}/'.format(timeString)
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    if not os.path.exists(summaryPath):
        os.makedirs(summaryPath)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #Start Queue Runner for Data Input
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    #Define Summary Writer
    mergedSummaries = tf.summary.merge([learningRateSummary, entropySummary, diceSummary])
    trainWriter = tf.summary.FileWriter(summaryPath + 'train/', sess.graph)
    evalWriter = tf.summary.FileWriter(summaryPath + 'eval/')
    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    accumulatedSteps = 0
    bestValdLoss = np.inf
    
    for i in range(cappedIterations):
        trainImages, trainMasks = trainDataSet.NextBatch(sess)
        feed_dict = {
            imagesPL: trainImages,
            labelsPL: trainMasks
        }
        
        if i % batchStepsBetweenSummaries == 0:
            _, trainingLoss, diceCoeff, imageSummaries = sess.run([trainOp, lossOp, diceOp, inputSummary], feed_dict=feed_dict)
            summaries = sess.run(mergedSummaries, feed_dict={lossPL: trainingLoss, dicePL: diceCoeff})
            trainWriter.add_summary(summaries, i)
            trainWriter.add_summary(imageSummaries, i)
            
            valdLoss, valdDice, valdImageSummaries = getPerformanceOnImages(sess, valdImages, valdMasks, imagesPL, labelsPL, lossOp, evalDiceOp, inputSummary)
            summaries = sess.run(mergedSummaries, feed_dict={lossPL: valdLoss, dicePL: valdDice})
            evalWriter.add_summary(summaries, i)
            evalWriter.add_summary(valdImageSummaries, i)
            print('Iteration {}, Training: [Loss = {}, Dice = {}], Validation: [Loss = {}, Dice = {}]'.format(i, trainingLoss, diceCoeff, valdLoss, valdDice))
            
            if valdLoss < bestValdLoss:
                bestValdLoss = valdLoss
                accumulatedSteps = 0
                saver.save(sess, modelPath)
                print('Model saved to path: {}'.format(modelPath))
            else:
                accumulatedSteps += batchStepsBetweenSummaries
                if accumulatedSteps > stepsBeforeStoppingCriteria:
                    print('Reached early stopping criteria with validation loss {}'.format(bestValdLoss))
                    break
        else:
            sess.run([trainOp], feed_dict=feed_dict)
    
    saver.restore(sess, modelPath)
    print('Model restored from path: {}'.format(modelPath))
    testLoss, testDice, _= getPerformanceOnImages(sess, testImages, testMasks, imagesPL, labelsPL, lossOp, evalDiceOp, None)
    print('Test: [Loss = {}, Dice = {}]'.format(testLoss, testDice))
    coord.request_stop()
    coord.join(threads)