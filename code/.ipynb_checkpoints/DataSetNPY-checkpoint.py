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
            augment='none'
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
        if self.augment != 'none':
            self.CreateAugmentOperations(augmentation=augment)

    def PreloadData(self):
        files = [x.encode() for x in self.filenames]
        self.loadedImages = np.reshape(self._loadImages(files), self.imageBatchDims).astype(np.float32)
        self.loadedLabels = np.reshape(self._loadLabels(files), self.labelBatchDims).astype(np.float32)
        self.maxItemsInQueue = 1
        self.preloaded = True
        
    def NextBatch(self, sess):
        if self.preloaded:
            return self.loadedImages, self.loadedLabels

        if self.augment == 'none':
            return sess.run([self.imageBatchOperation, self.labelBatchOperation])
        else:
            return sess.run([self.augmentedImageOperation, self.labelBatchOperation])

    def GetBatchOperations(self):
        return self.imageBatchOperation, self.labelBatchOperation

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

    def CreateAugmentOperations(self, augmentation='flip'):
        with tf.variable_scope('DataAugmentation'):
            if augmentation == 'flip':
                augmentedImageOperation = tf.reverse(self.imageBatchOperation,
                                                     axis=[1],
                                                     name='flip')
            elif augmentation == 'translate':
                imageRank = 3
                maxPad = 6
                minPad = 0
                randomPadding = tf.random_uniform(shape=(3, 2),
                                                  minval=minPad,
                                                  maxval=maxPad + 1,
                                                  dtype=tf.int32)
                randomPadding = tf.pad(randomPadding, paddings=[[1, 1], [0, 0]])
                paddedImageOperation = tf.pad(self.imageBatchOperation, randomPadding)
                sliceBegin = randomPadding[:, 1]
                sliceEnd = self.imageBatchDims
                augmentedImageOperation = tf.slice(paddedImageOperation,
                                                sliceBegin,
                                                sliceEnd)

            chooseOperation = tf.cond(
                tf.equal(
                    tf.ones(shape=(), dtype=tf.int32),
                    tf.random_uniform(shape=(), dtype=tf.int32, minval=0, maxval=2)
                ),
                lambda: augmentedImageOperation,
                lambda: self.imageBatchOperation,
                name='ChooseAugmentation'
            )
            self.augmentedImageOperation = tf.reshape(chooseOperation, self.imageBatchDims)

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