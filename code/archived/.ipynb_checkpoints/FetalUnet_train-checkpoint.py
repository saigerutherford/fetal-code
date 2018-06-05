import time
import os

from medpy.io import load, save
import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d

import tensorflow as tf

#import radio

lossfunc = 1
#pretraining on BET images (0) or training on hand labeled (1)
train = 2

folders = ["bet","labeled","small"]

# Network Parameters
tf.reset_default_graph()
width = 96
height = 96
n_channels = 1
n_classes = 2 # total classes (brain, non-brain)
x = tf.placeholder(tf.float32, [None, width, height, n_channels])
y_ = tf.placeholder(tf.float32, [None, n_classes])
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.00001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.9, staircase=True)

################Create Model######################
conv1 = conv_2d(x, 32, 3, activation='relu', padding='same', regularizer="L2")
conv1 = conv_2d(conv1, 32, 3, activation='relu', padding='same', regularizer="L2")
pool1 = max_pool_2d(conv1, 2)

conv2 = conv_2d(pool1, 64, 3, activation='relu', padding='same', regularizer="L2")
conv2 = conv_2d(conv2, 64, 3, activation='relu', padding='same', regularizer="L2")
pool2 = max_pool_2d(conv2, 2)

conv3 = conv_2d(pool2, 128, 3, activation='relu', padding='same', regularizer="L2")
conv3 = conv_2d(conv3, 128, 3, activation='relu', padding='same', regularizer="L2")
pool3 = max_pool_2d(conv3, 2)

conv4 = conv_2d(pool3, 256, 3, activation='relu', padding='same', regularizer="L2")
conv4 = conv_2d(conv4, 256, 3, activation='relu', padding='same', regularizer="L2")
pool4 = max_pool_2d(conv4, 2)

conv5 = conv_2d(pool4, 512, 3, activation='relu', padding='same', regularizer="L2")
conv5 = conv_2d(conv5, 512, 3, activation='relu', padding='same', regularizer="L2")

up6 = upsample_2d(conv5,2)
up6 = tflearn.layers.merge_ops.merge([up6, conv4], 'concat',axis=3)
conv6 = conv_2d(up6, 256, 3, activation='relu', padding='same', regularizer="L2")
conv6 = conv_2d(conv6, 256, 3, activation='relu', padding='same', regularizer="L2")

up7 = upsample_2d(conv6,2)
up7 = tflearn.layers.merge_ops.merge([up7, conv3],'concat', axis=3)
conv7 = conv_2d(up7, 128, 3, activation='relu', padding='same', regularizer="L2")
conv7 = conv_2d(conv7, 128, 3, activation='relu', padding='same', regularizer="L2")

up8 = upsample_2d(conv7,2)
up8 = tflearn.layers.merge_ops.merge([up8, conv2],'concat', axis=3)
conv8 = conv_2d(up8, 64, 3, activation='relu', padding='same', regularizer="L2")
conv8 = conv_2d(conv8, 64, 3, activation='relu', padding='same', regularizer="L2")

up9 = upsample_2d(conv8,2)
up9 = tflearn.layers.merge_ops.merge([up9, conv1],'concat', axis=3)
conv9 = conv_2d(up9, 32, 3, activation='relu', padding='same', regularizer="L2")
conv9 = conv_2d(conv9, 32, 3, activation='relu', padding='same', regularizer="L2")

pred = conv_2d(conv9, 2, 1,  activation='linear', padding='valid')

dynamic_shape = tf.shape(pred)
pred_reshape = tf.reshape(pred, [dynamic_shape[0]*width*height, n_classes])

#class_counts = tf.divide(1,tf.reduce_sum(y_, 0))
#sample_weights = tf.gather(class_counts, tf.cast(tf.argmax(y_,1),tf.int32))

#loss = tf.losses.softmax_cross_entropy(y_, pred_reshape, weights=sample_weights)
def tversky_loss(labels, predictions, alpha=0.3, beta=0.7, smooth=1e-10):
    labels = tf.contrib.layers.flatten(labels)
    predictions = tf.contrib.layers.flatten(predictions)
    truepos = tf.reduce_sum(labels * predictions)
    fp_and_fn = (alpha * tf.reduce_sum(predictions * (1 - labels))
                 + beta * tf.reduce_sum((1 - predictions) * labels))

    return -(truepos + smooth) / (truepos + smooth + fp_and_fn)

#mask = tf.reshape(1-tf.argmax(y_,1),[-1,1])
#mask = tf.gather(tf.constant([0.,1.]),mask)
mask = tf.slice(y_,[0,0],[-1,1])
#predictions = 0.5*(tf.sign(tf.slice(tf.nn.softmax(pred_reshape),[0,0],[-1,1])-0.5-1e-10)) + 0.5
predictions = tf.slice(tf.nn.softmax(pred_reshape),[0,0],[-1,1])

predbinary = tf.round(tf.slice(tf.nn.softmax(pred_reshape),[0,0],[-1,1]))
dice = (2*tf.reduce_sum(mask*predbinary))/(tf.reduce_sum(mask)+tf.reduce_sum(predbinary))
masksum = tf.reduce_sum(mask)
predsum = tf.reduce_sum(predbinary)

#predictions = tf.reshape(1-tf.argmax(tf.nn.softmax(pred_reshape),1),[-1,1])
#predictions = tf.gather(tf.constant([0.,1.]),predictions)

if (lossfunc==0):
    weights = tf.reduce_max(tf.divide(y_,tf.reduce_sum(y_,0)*2),1)
    loss = tf.losses.softmax_cross_entropy(y_,pred_reshape,weights=weights,label_smoothing=0.1)
elif (lossfunc==1):
    loss = tversky_loss(mask,predictions,alpha=0.3,beta=0.7)

train_step = (
    tf.train.AdamOptimizer(learning_rate)
    .minimize(loss, global_step=global_step)
)


###############Initialize Model#######################
init = tf.initialize_all_variables()
    
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()
#Load model
model_path = "../Model/Fetal_2D_Ref_6980_norm0.ckpt"
model_path = "../Model/Fetal_2D_functional.ckpt"
if (train==1):
    #saver.restore(sess, model_path)
    model_path = "../Model/Fetal_2D_functional_labeled.ckpt"
if (train==2):
    model_path = "../Model/Fetal_2D_functional_small.ckpt"

##############Evaluate Model on data#############
loops = 1
t = time.time()
while True:
    
    inputCounter = 0
    for f in os.walk('../TrainingData'):
        inputCounter += 1
        if inputCounter > 1:
            if folders[train] in f[0]:
                for f2 in os.listdir(f[0]): 
                    if not("mask") in f2 and "nii" in f2:
                        #t = time.time()
                        image_data, image_header = load(f[0]+'/'+f2) # Load data
                        image_data = (image_data - np.mean(image_data))/np.std(image_data)
                        imageDim = np.shape(image_data)   
                        image_data = np.swapaxes(image_data,0,2) # Bring the last dim to the first
                        image_data = np.swapaxes(image_data,1,2) # Bring the last dim to the first
                        input_data = image_data[..., np.newaxis] # Add one axis to the end
                        #input_dataf = input_data
                        input_dataf = np.concatenate((input_data, np.flip(input_data,1), np.flip(input_data,2), np.flip(np.flip(input_data,1),2)),axis=0)

                        label_data, label_header = load(f[0]+'/mask_'+f2)
                        label_data = np.swapaxes(label_data,0,2)
                        label_data = np.swapaxes(label_data,1,2)
                        label_data = label_data[...,np.newaxis]
                        label_data = np.where(label_data>0, 1, 0)
                        #label_dataf = label_data
                        label_dataf = np.concatenate((label_data, np.flip(label_data,1), np.flip(label_data,2), np.flip(np.flip(label_data,1),2)), axis=0)
                        label_dataf = np.reshape(label_dataf,(imageDim[2]*width*height*4,1))
                        label_dataf = np.concatenate((label_dataf, 1-label_dataf),axis=1)
                        result = sess.run(train_step, feed_dict={x: input_dataf, y_: label_dataf})
                        out = sess.run(dice,feed_dict={x: input_dataf, y_:label_dataf})
                        print(out)
                        a = tf.identity(global_step)
                        b = sess.run(a)
                        if (np.mod(b,50)==0):
                            print(b)
                            saver.save(sess,model_path)
                            elapsed = time.time() - t 
                            print(elapsed)
                            #t = time.time()
    print(loops)
    loops+=1

