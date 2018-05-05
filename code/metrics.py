import tensorflow as tf

def Dice(binaryLabels, binaryOutputLayer):
    with tf.variable_scope('DiceCoefficient'):
        binaryLabelsSum = tf.reduce_sum(binaryLabels, axis=[1,2])
        binaryOutputSum = tf.reduce_sum(binaryOutputLayer, axis=[1,2])
        eps = tf.constant(0.0000001, dtype=tf.float32, shape=[1, 1])
        denom = tf.cast(binaryLabelsSum + binaryOutputSum, tf.float32) + eps
        numer = tf.cast(2 * tf.reduce_sum(binaryOutputLayer * binaryLabels, axis=[1,2]), tf.float32) + eps
        diceOp = tf.reduce_mean(numer / denom)
    return diceOp

def PatientDice(binaryLabels, binaryOutputLayer):
    """
    Calculates the dice coefficient on a per-patient basis, assuming 
    that the output is (37, 96, 96, 1) where 37 represents a spatial dimension
    """
    with tf.variable_scope('DiceCoefficient'):
        binaryLabelsSum = tf.reduce_sum(binaryLabels)
        binaryOutputSum = tf.reduce_sum(binaryOutputLayer)
        eps = tf.constant(0.0000001, dtype=tf.float32, shape=())
        denom = tf.cast(binaryLabelsSum + binaryOutputSum, tf.float32) + eps
        numer = tf.cast(2 * tf.reduce_sum(binaryOutputLayer * binaryLabels), tf.float32) + eps
        diceOp = numer / denom
    return diceOp