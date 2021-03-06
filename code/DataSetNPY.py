import tensorflow as tf
import numpy as np

class DataSetNPY(object):
    """
    This class supports reading in batches of .npy data.
    It is less efficient than converting your data into TFRecords (but
    it is difficult to convert 3D data into such a format), and it is also
    less efficient than converting your data into byte format
    (which is possible for 3D data, but runs into computational issues
    regarding model building). 
    """
    def __init__(self,
            filenames,
            imageBaseString,
            imageBatchDims,
            labelBatchDims,
            labelBaseString,
            batchSize=64,
            maxItemsInQueue=100,
            shuffle=True,
            augment=None
        ):
        self.filenames = filenames
        self.batchSize = batchSize
        self.imageBatchDims = imageBatchDims
        self.labelBatchDims = labelBatchDims
        self.imageBaseString = imageBaseString
        self.labelBaseString = labelBaseString
        self.maxItemsInQueue = maxItemsInQueue
        self.preloaded=False
        self.loadedImages=None
        self.loadedLabels=None
        stringQueue = tf.train.string_input_producer(filenames, shuffle=shuffle, capacity=maxItemsInQueue)
        dequeueOp = stringQueue.dequeue_many(batchSize)
        self.dequeueOp = dequeueOp
        self.imageBatchOperation = tf.reshape(
            tf.py_func(self._loadImages, [dequeueOp], tf.float32),
            imageBatchDims)
        self.labelBatchOperation = tf.reshape(
            tf.py_func(self._loadLabels, [dequeueOp], tf.float32),
            labelBatchDims)
        self.augment = augment
        if self.augment != None:
            self.CreateAugmentOperations()

    def PreloadData(self):
        files = [x.encode() for x in self.filenames]
        self.loadedImages = np.reshape(self._loadImages(files), self.imageBatchDims).astype(np.float32)
        self.loadedLabels = np.reshape(self._loadLabels(files), self.labelBatchDims).astype(np.float32)
        self.maxItemsInQueue = 1
        self.preloaded = True
        
    def NextBatch(self, sess):
        if self.preloaded:
            return self.loadedImages, self.loadedLabels

        if self.augment is None:
            return sess.run([self.imageBatchOperation, self.labelBatchOperation])
        else:
            return sess.run([self.augmentedImageOperation, self.augmentedLabelOperation])

    def GetBatchOperations(self):
        if self.augment is None:
            return self.imageBatchOperation, self.labelBatchOperation
        else:
            return self.augmentedImageOperation, self.augmentedLabelOperation

    def GetRandomBatchOperations(self):
        randomIndexOperation = tf.random_uniform(shape=(self.batchSize,),
                                                dtype=tf.int32,
                                                minval=0,
                                                maxval=len(self.filenames))
        filenameTensor = tf.constant(self.filenames, dtype=tf.string)
        randomFilenames = tf.gather(filenameTensor, randomIndexOperation)
        randomImageBatch = tf.reshape(
            tf.py_func(self._loadImages, [randomFilenames], tf.float32),
            self.imageBatchDims)
        randomLabelBatch = tf.reshape(
            tf.py_func(self._loadLabels, [randomFilenames], tf.float32),
            self.labelBatchDims)
        return randomImageBatch, randomLabelBatch

    def returnCoinPred(self):
        return tf.equal(
                    tf.ones(shape=(), dtype=tf.int32),
                    tf.random_uniform(shape=(), dtype=tf.int32, minval=0, maxval=2)
                )
    
    def randomTranslation(self, imageTensor, maskTensor):
        maxPad = 20
        minPad = 0
        randomPadding = tf.random_uniform(shape=(2,2),
                                          minval=minPad,
                                          maxval=maxPad + 1,
                                          dtype=tf.int32)
        randomPadding = tf.pad(randomPadding, paddings=[[0,1], [0,0]])
        paddedImageOperation = tf.pad(imageTensor, randomPadding)
        paddedMaskOperation = tf.pad(maskTensor, randomPadding)
        sliceBegin = randomPadding[:, 1]
        sliceEnd = [self.imageBatchDims[1], self.imageBatchDims[2], self.imageBatchDims[0]]
        augmentedImageOperation = tf.slice(paddedImageOperation,
                                           sliceBegin,
                                           sliceEnd)
        augmentedMaskOperation = tf.slice(paddedMaskOperation,
                                          sliceBegin,
                                          sliceEnd)
        return augmentedImageOperation, augmentedMaskOperation
        
    def chooseTensor(self, tensorAImage, tensorBImage, tensorALabel, tensorBLabel):
        pred = self.returnCoinPred()
        return tf.cond(pred, lambda: tensorAImage, lambda: tensorBImage), tf.cond(pred, lambda: tensorALabel, lambda: tensorBLabel)
        
    def CreateAugmentOperations(self):
        with tf.variable_scope('DataAugmentation'):
            squeezedImages = tf.reshape(self.imageBatchOperation, shape=[self.imageBatchDims[0], self.imageBatchDims[1], self.imageBatchDims[2]])
            squeezedImages = tf.transpose(squeezedImages, [1, 2, 0])
            squeezedLabels = tf.transpose(self.labelBatchOperation, [1, 2, 0])
            
            with tf.variable_scope('FlipUpDown'):
                flipUpDownImage = tf.image.flip_up_down(squeezedImages)
                flipUpDownLabel = tf.image.flip_up_down(squeezedLabels)
                self.augmentedImageOperation, self.augmentedLabelOperation = self.chooseTensor(flipUpDownImage, squeezedImages,
                                                                                               flipUpDownLabel, squeezedLabels)
            
            with tf.variable_scope('FlipLeftRight'):
                flipLeftRightImage = tf.image.flip_left_right(self.augmentedImageOperation)
                flipLeftRightLabel = tf.image.flip_left_right(self.augmentedLabelOperation)
                self.augmentedImageOperation, self.augmentedLabelOperation = self.chooseTensor(flipLeftRightImage, self.augmentedImageOperation,
                                                            flipLeftRightLabel, self.augmentedLabelOperation)
            with tf.variable_scope('Rotations'):
                for i in range(3):
                    randomRotateImage = tf.image.rot90(self.augmentedImageOperation)
                    randomRotateLabel = tf.image.rot90(self.augmentedLabelOperation)
                    self.augmentedImageOperation, self.augmentedLabelOperation = self.chooseTensor(randomRotateImage, self.augmentedImageOperation,
                                                                randomRotateLabel, self.augmentedLabelOperation)
            
            with tf.variable_scope('Translations'):
                self.augmentedImageOperation, self.augmentedLabelOperation = self.randomTranslation(self.augmentedImageOperation,
                                                                                    self.augmentedLabelOperation)
            
            self.augmentedImageOperation = tf.reshape(tf.transpose(self.augmentedImageOperation, [2, 0, 1]), self.imageBatchDims)
            self.augmentedLabelOperation = tf.reshape(tf.transpose(self.augmentedLabelOperation, [2, 0, 1]), self.labelBatchDims)

    def _loadImages(self, x):
        images = []
        for name in x:
            images.append(np.load('{}{}'.format(self.imageBaseString, name.decode('utf-8'))).astype(np.float32))
        images = np.array(images)
        return images

    def _loadLabels(self, x):
        labels = []
        for name in x:
            labels.append(np.load('{}{}'.format(self.labelBaseString, name.decode('utf-8'))).astype(np.float32))
        labels = np.array(labels)
        return labels

if __name__ == '__main__':
    import os
    # Global Values
    width = 96
    height = 96
    n_channels = 1
    n_classes = 2

    dataBaseString = '/scratch/wiensj_fluxg/psturm/FETAL/'
    trainPrefix = 'trainImages/'
    labelPrefix = 'trainMasks/mask_'
    filenames = os.listdir('{}{}'.format(dataBaseString, trainPrefix))
    imageBatchDims = (-1, width, height, n_channels)
    labelBatchDims = (-1, width, height, n_channels)

    trainDataSet = DataSetNPY(filenames=filenames,
                              imageBaseString='{}{}'.format(dataBaseString, trainPrefix),
                              imageBatchDims=imageBatchDims,
                              labelBatchDims=labelBatchDims,
                              labelBaseString='{}{}'.format(dataBaseString, labelPrefix),
                              batchSize=64)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        images, labels = trainDataSet.NextBatch(sess)  
        print(images.shape)
        print(labels.shape)
        
        coord.request_stop()
        coord.join(threads)