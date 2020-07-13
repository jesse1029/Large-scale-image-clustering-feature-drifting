# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

n_classes = 10 # 标签的维度
dropout = 0.5 # Dropout 的概率

# 存储所有的网络参数
ke=[64,128, 128, 1000]
weights = {
	'wc1': tf.Variable(tf.random_normal([5, 5, 1, ke[0]], stddev=0.01)),
	'wc1r': tf.Variable(tf.random_normal([5, 5, ke[0], ke[1]], stddev=0.01)),
	
	'wc2': tf.Variable(tf.random_normal([5, 5, ke[0], ke[1]], stddev=0.01)),
	'wc9': tf.Variable(tf.random_normal([4*4*ke[1], ke[2]], stddev=0.01)),
	'wc10': tf.Variable(tf.random_normal([ke[2], ke[3]], stddev=0.01)),
	'wc9r': tf.Variable(tf.random_normal([4*4*ke[2], 28*28], stddev=0.01)),
	
	'out': tf.Variable(tf.random_normal([ke[3], n_classes], stddev=0.01))
}
biases = {
	'bc1': tf.Variable(tf.random_normal([ke[0]], mean=0.1, stddev=0.01)),
	'bc1r': tf.Variable(tf.random_normal([ke[0]], mean=0.1, stddev=0.01)),
	
	'bc2': tf.Variable(tf.random_normal([ke[1]], mean=0.1, stddev=0.01)),
	'bc9': tf.Variable(tf.random_normal([ke[2]], mean=0.1, stddev=0.01)),
	'bc10': tf.Variable(tf.random_normal([ke[3]], mean=0.1, stddev=0.01)),
	'bc9r': tf.Variable(tf.random_normal([28*28], mean=0.1, stddev=0.01)),
	
	'out': tf.Variable(tf.random_normal([n_classes], mean=0.1, stddev=0.01))
}


def batch_norm(x, n_out, scope='bn'):
	"""
	Batch normalization on convolutional maps.
	Args:
		x:           Tensor, 4D BHWD input maps
		n_out:       integer, depth of input maps
		phase_train: boolean tf.Varialbe, true indicates training phase
		scope:       string, variable scope
	Return:
		normed:      batch-normalized maps
	"""

	with tf.variable_scope(scope):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
									 name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
									  name='gamma', trainable=True)

		batch_mean, batch_var = tf.nn.moments(x, [0,1,2])
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = mean_var_with_update()
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed

# 卷积操作
def conv2d(name, l_input, w, b, s, p):
	n_out=l_input.get_shape().as_list()[3]
	return tf.nn.bias_add(tf.nn.conv2d(tf.nn.relu(batch_norm(l_input, n_out), name=name), w, strides=[1,s,s,1], padding=p),b)

def conv2dwobn(name, l_input, w, b, s, p):
	n_out=w.get_shape().as_list()[3]
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1,s,s,1], padding=p),b), name=name)
	
# 最大下采样操作
def max_pool(name, l_input, k, s):
	return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)

# 归一化操作
def norm(name, l_input, lsize=4):
	return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
    

# 定义整个网络 
def lenet_ccnn(_X,  _dropout):
	global weights
	global biases
	
	conv1 = conv2d('conv1', _X, weights['wc1'], biases['bc1'], 1, 'VALID')
	pool1 = max_pool('pool1', conv1, k=2,s=2)	
	conv2 = conv2d('conv2', pool1, weights['wc2'], biases['bc2'], 1, 'VALID')
	pool2 = max_pool('pool2', conv2, k=2,s=2)	
	pool2 = tf.reshape(pool2, [-1, weights['wc9'].get_shape().as_list()[0]])
	
	
	print "conv1:" + str(conv1.get_shape().as_list())
	print "pool1:" + str(pool1.get_shape().as_list())
	print "conv2:" + str(conv2.get_shape().as_list())
	print "pool2:" + str(pool2.get_shape().as_list())
	print "pool2-reshape:" + str(pool2.get_shape().as_list())
	
	fca=tf.matmul(pool2, weights['wc9']) + biases['bc9']
	fca= tf.nn.relu(fca)
	fca = tf.nn.dropout(fca, _dropout)
	
	fcb=tf.matmul(fca, weights['wc10']) + biases['bc10']
	fcb= tf.nn.relu(fcb)
	fcb = tf.nn.dropout(fcb, _dropout)
	
	out = tf.nn.softmax(tf.matmul(fcb, weights['out']) + biases['out'], name='fc3') # Relu activation
	return fcb, out


def lenet_ccnn_rec(_X, _dropout):
	global weights
	global biases
	
	conv1 = conv2d('conv1', _X, weights['wc1'], biases['bc1'], 1, 'VALID')
	pool1 = max_pool('pool1', conv1, k=2,s=2)	
	conv2 = conv2d('conv2', pool1, weights['wc2'], biases['bc2'], 1, 'VALID')
	pool2 = max_pool('pool2', conv2, k=2,s=2)	
	pool2 = tf.reshape(pool2, [-1, weights['wc9'].get_shape().as_list()[0]])

	fcb=tf.matmul(pool2, weights['wc9r']) + biases['bc9r']
	out=tf.reshape(fcb,[-1,28,28,1])
	return out


