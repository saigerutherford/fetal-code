import time
import os

from medpy.io import load, save
import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d

import tensorflow as tf

#testing of pretraining on BET images (0) or training on hand labeled (1)
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

mask = tf.slice(y_,[0,0],[-1,1])
predictions = tf.slice(tf.nn.softmax(pred_reshape),[0,0],[-1,1])

predbinary = tf.round(tf.slice(tf.nn.softmax(pred_reshape),[0,0],[-1,1]))
dice = (2*tf.reduce_sum(mask*predbinary))/(tf.reduce_sum(mask)+tf.reduce_sum(predbinary))



###############Initialize Model#######################
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()
#Load model
model_path = "../Model/Fetal_2D_Ref_6980_norm0.ckpt"
model_path = "../Model/Fetal_2D_functional.ckpt" #our functional model
if (train==1):
    model_path = "../Model/Fetal_2D_functional_labeled.ckpt"
if (train==2):
    model_path = "../Model/Fetal_2D_functional_small.ckpt"

saver.restore(sess, model_path)

##############Evaluate Model on data#############
inputCounter = 0
for f in os.walk('../TestingData'):
    inputCounter += 1
    if inputCounter > 1:
        if folders[train] in f[0]:
            print(f[0])
            for f2 in os.listdir(f[0]): 
                if not("mask") in f2 and "nii" in f2:
                    t = time.time()
                    image_data, image_header = load(f[0]+'/'+f2) # Load data
                    image_data = (image_data - np.mean(image_data))/np.std(image_data)
                    imageDim = np.shape(image_data)   
                    image_data = np.swapaxes(image_data,0,2) # Bring the last dim to the first
                    image_data = np.swapaxes(image_data,1,2) # Bring the last dim to the first
                    input_data = image_data[..., np.newaxis] # Add one axis to the end
                    #label_data, label_header = load(f[0]+'/mask_'+f2)
                    #label_data = np.swapaxes(label_data,0,2)
                    #label_data = np.swapaxes(label_data,1,2)
                    #label_data = label_data[...,np.newaxis]
                    #label_data = np.where(label_data>0, 1, 0)
                    #label_data = np.reshape(label_data,(imageDim[2]*width*height,1))
                    #label_data = np.concatenate((label_data,1-label_data),axis=1)

                    out = sess.run(tf.nn.softmax(pred_reshape), feed_dict={x: input_data}) # Find probabilities
                    _out = np.reshape(out, (imageDim[2], width, height, n_classes)) # Reshape to input shape
                    #out = sess.run(dice,feed_dict={x: input_data, y_:label_data})
                    #print(out)

                    mask = 1 - np.argmax(np.asarray(_out), axis=3).astype(float) # Find mask
                    mask = np.swapaxes(mask, 1, 2) # Bring the first dim to the last
                    mask = np.swapaxes(mask, 0, 2) # Bring the first dim to the last
                    
                    #elapsed = time.time() - t 
                    save(mask, f[0]+'/Net_mask_'+f2, image_header) # Save the mask
                    
                    elapsed = time.time() - t
                    print(elapsed)
                    
