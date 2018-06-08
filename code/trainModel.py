import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime

from buildModel import CNN
from dataHelper import GetDataSets
from DataSetNPY import DataSetNPY
from metrics import *
from globalVars import *

def getPatientLosses(sess, images, masks, imagesPL, labelsPL, lossOp):
    patientLosses = np.zeros((images.shape[-1]))
    
    for i in range(images.shape[-1]):
        patientImages = np.reshape(images[:, :, :, i], (37, 96, 96, 1))
        patientMasks = np.reshape(masks[:, :, :, i], (37, 96, 96))
        feed_dict = {
            imagesPL: patientImages,
            labelsPL: patientMasks
        }
        loss = sess.run(lossOp, feed_dict=feed_dict)
        patientLosses[i] = loss
    
    return patientLosses

def getBootstrapPerformanceOnImages(sess, images, masks, imagesPL, labelsPL, lossOp):
    patientLosses = getPatientLosses(sess, images, masks, imagesPL, labelsPL, lossOp)
        
    accumulatedLosses = []
    for i in range(1000):
        accumulatedLosses.append(np.mean(np.random.choice(patientLosses, size=images.shape[-1], replace=True)))
    accumulatedLosses = np.sort(accumulatedLosses)
    lower = accumulatedLosses[25]
    mid = np.mean(patientLosses)
    upper = accumulatedLosses[975]
    
    return lower, mid, upper
        
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
            summaryDict = {
                imagesPL: np.reshape(patientImages[18, :, :, :], (1, 96, 96, 1)),
                labelsPL: np.reshape(patientMasks[18, :,  :], (1, 96, 96))
            }
            valdInputSummary = sess.run(inputSummary, feed_dict=summaryDict)
        loss, dice = sess.run([lossOp, diceOp], feed_dict=feed_dict)
        accumulatedLoss += loss
        accumulatedDice += dice
        denom += 1
    return accumulatedLoss / denom, accumulatedDice / denom, valdInputSummary

def main(train=True, timeString=None):
    #Define Dataset Operations
    trainDataSet, valdDataSet, testDataSet = GetDataSets()
    imageShape = [None, height, width, n_channels]
    maskShape = [None, height, width]
    with tf.variable_scope('InputPlaceholders'):
        imagesPL = tf.placeholder(dtype=tf.float32, shape=imageShape, name='imagesPL')
        labelsPL = tf.placeholder(dtype=tf.float32, shape=maskShape, name='masksPL')
        binaryLabels = tf.cast(labelsPL > 0, tf.int32)
    valdImages, valdMasks = valdDataSet.NextBatch(None) # Of shape (37, 96, 96, -1)
    valdImages = np.swapaxes(valdImages, 0, 3)
    valdMasks = np.swapaxes(valdMasks, 0, 3) 
    testImages, testMasks = testDataSet.NextBatch(None)
    testFileNames = testDataSet.filenames
    testImages = np.swapaxes(testImages, 0, 3)
    testMasks = np.swapaxes(testMasks, 0, 3)
    
    # Define Learning Parameters
    globalStep = tf.Variable(0, trainable=False)
    initLearningRate = 0.0001
    learningRate = tf.train.exponential_decay(initLearningRate, globalStep,
                                               10000, 0.9, staircase=True)
    learningRateSummary = tf.summary.scalar('learningRate', learningRate)

    #Build Model
    outputLayer = CNN(imagesPL)
    binaryOutputLayer = tf.argmax(outputLayer, axis=3, output_type=tf.int32)
    inputSummary = tf.summary.merge([tf.summary.image('Images', imagesPL, max_outputs=1), 
                                     tf.summary.image('TrueMasks', tf.expand_dims(tf.cast(binaryLabels, tf.float32), -1),  max_outputs=1),
                                     tf.summary.image('PredictedMasks', tf.expand_dims(tf.cast(binaryOutputLayer, tf.float32), -1), max_outputs=1)
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
        
        with tf.variable_scope('Accuracy'):
            accuracyOp = tf.reduce_sum(tf.cast(tf.equal(binaryOutputLayer, binaryLabels), tf.float32)) / tf.cast(tf.reduce_prod(tf.shape(binaryLabels)), tf.float32)
        with tf.variable_scope('TruePositive'):
            numer = tf.reduce_sum(tf.cast(binaryLabels, tf.float32) * tf.cast(tf.equal(binaryOutputLayer, binaryLabels), tf.float32))
            denom = tf.reduce_sum(tf.cast(binaryLabels, tf.float32))
            truePositiveOp =  numer / denom 
        with tf.variable_scope('TrueNegative'):
            trueNegativeOp = tf.reduce_sum(tf.cast(1 - binaryLabels, tf.float32) * tf.cast(tf.equal(binaryOutputLayer, binaryLabels), tf.float32)) / tf.reduce_sum(tf.cast(1 - binaryLabels, tf.float32))

    #Initialize weights, sessions
    if not train and timeString is None:
        raise ValueError('Specified train=False but timeString was not provided. Specify a timeString to load a checkpoint from.')
    elif train:
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
        bestDice = 0
        
        if train:
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

                    if valdDice > bestDice:
                        bestDice = valdDice
                        accumulatedSteps = 0
                        saver.save(sess, modelPath)
                        print('Model saved to path: {}'.format(modelPath))
                    else:
                        accumulatedSteps += batchStepsBetweenSummaries
                        if accumulatedSteps > stepsBeforeStoppingCriteria:
                            print('Reached early stopping criteria with validation dice {}'.format(bestDice))
                            break
                else:
                    sess.run([trainOp], feed_dict=feed_dict)

        saver.restore(sess, modelPath)
        print('Model restored from path: {}'.format(modelPath))
        lower, mid, upper = getBootstrapPerformanceOnImages(sess, testImages, testMasks, imagesPL, labelsPL, lossOp)
        print('Test: Loss = {}, ({}, {})'.format(mid, lower, upper))
        lower, mid, upper = getBootstrapPerformanceOnImages(sess, testImages, testMasks, imagesPL, labelsPL, evalDiceOp)
        print('Test: DICE = {}, ({}, {})'.format(mid, lower, upper))
        lower, mid, upper = getBootstrapPerformanceOnImages(sess, testImages, testMasks, imagesPL, labelsPL, accuracyOp)
        print('Test: Accuracy = {}, ({}, {})'.format(mid, lower, upper))
        lower, mid, upper = getBootstrapPerformanceOnImages(sess, testImages, testMasks, imagesPL, labelsPL, truePositiveOp)
        print('Test: TruePositive = {}, ({}, {})'.format(mid, lower, upper))
        lower, mid, upper = getBootstrapPerformanceOnImages(sess, testImages, testMasks, imagesPL, labelsPL, trueNegativeOp)
        print('Test: TrueNegative = {}, ({}, {})'.format(mid, lower, upper))
        
        df = pd.DataFrame(data =  {
            'Volume': testFileNames,
            'Dice': getPatientLosses(sess, testImages, testMasks, imagesPL, labelsPL, evalDiceOp),
            'Accuracy': getPatientLosses(sess, testImages, testMasks, imagesPL, labelsPL, accuracyOp),
            'Sensitivity': getPatientLosses(sess, testImages, testMasks, imagesPL, labelsPL, truePositiveOp),
            'Specificty': getPatientLosses(sess, testImages, testMasks, imagesPL, labelsPL, trueNegativeOp)
        })
        df.to_csv('PatientMetrics.csv', index=False)
        
        coord.request_stop()
        coord.join(threads)
    

#This line evaluates the model on the test set. It does not train anything.
main(train=False, timeString='2018-06-07_14:07') 

#This line trains a model from scratch on the training set.
#main()

