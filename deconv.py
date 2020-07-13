# -*- coding: utf-8 -*-

""" AlexNet.
Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import cv2
import os
import tensorflow as tf

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression



# Building 'AlexNet'
network = input_data(shape=[None, 256, 256, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 1000, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
   
fn='/Data/jess/ILSVRC2012_img_val/'
fn2='/home/jess/caffe-old/data/ilsvrc12/val.txt'

with open(fn2) as fx:
	content = fx.readlines()
	
size = 256, 256
skip_step = 10
len1 = len(content)
data = np.zeros((len1/skip_step,256,256,3))
label1 = np.zeros((len1/skip_step, 1000));

cnt = 0
for k in range(0,len1-1,skip_step):
	fn1 = content[k].replace('\n','')
	fn2 = fn1.split(' ')

	fnx = fn+fn2[0]
	fnx = fnx.replace(" ","")
	#print(fnx)
	im = cv2.imread(fnx)
	#print(type(im))
	im = np.asarray(im)
	
	im = cv2.resize(im, (256,256))
	row,col,channels = im.shape
	if k%500==0:
		print("Processing " + str(k) + " images / " + str(len1) + " images.")
		print(str(k )+ ' image where size is ' + str(col) + "x" +str(row))
		
	try:
		label1[cnt, int(fn2[1])] = 1
		data[cnt,:,:,:]=im
		cnt = cnt + 1
	except Exception:
		pass
		

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet', max_checkpoints=1, tensorboard_verbose=2)
model.fit(data, label1, n_epoch=1000, validation_set=0.1, shuffle=True, show_metric=True, batch_size=64, snapshot_step=200, snapshot_epoch=False, run_id='alexnet_imagenet_small')
