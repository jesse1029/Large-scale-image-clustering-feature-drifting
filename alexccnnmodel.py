# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

n_classes = 1000 # 标签的维度
dropout = 0.5 # Dropout 的概率

# 存储所有的网络参数

weights = {
	'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=0.01)),
	'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=0.01)),
	'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=0.01)),
	'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384], stddev=0.01)),
	'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=0.01)),
	'wc6': tf.Variable(tf.random_normal([3, 3, 256, 6144], stddev=0.01)),
	'wc7': tf.Variable(tf.random_normal([1, 1, 6144, 6144], stddev=0.01)),
	'wc8': tf.Variable(tf.random_normal([3, 3, 6144, 2048], stddev=0.01)),
	'wc9': tf.Variable(tf.random_normal([3, 3, 2048, n_classes], stddev=0.01)),
	'out': tf.Variable(tf.random_normal([n_classes, n_classes], stddev=0.01))
}
biases = {
	'bc1': tf.Variable(tf.random_normal([96], mean=0.1, stddev=0.01)),
	'bc2': tf.Variable(tf.random_normal([256], mean=0.1, stddev=0.01)),
	'bc3': tf.Variable(tf.random_normal([384], mean=0.1, stddev=0.01)),
	'bc4': tf.Variable(tf.random_normal([384], mean=0.1, stddev=0.01)),
	'bc5': tf.Variable(tf.random_normal([256], mean=0.1, stddev=0.01)),
	'bc6': tf.Variable(tf.random_normal([6144], mean=0.1, stddev=0.01)),
	'bc7': tf.Variable(tf.random_normal([6144], mean=0.1, stddev=0.01)),
	'bc8': tf.Variable(tf.random_normal([2048], mean=0.1, stddev=0.01)),
	'bc9': tf.Variable(tf.random_normal([n_classes], mean=0.1, stddev=0.01)),
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
	n_out=w.get_shape().as_list()[3]
	return tf.nn.relu(norm(name+'n', batch_norm(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1,s,s,1], padding=p),b), n_out)), name=name)

def conv2dwobn(name, l_input, w, b, s, p):
	n_out=w.get_shape().as_list()[3]
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1,s,s,1], padding=p),b), name=name)
	
# 最大下采样操作
def max_pool(name, l_input, k, s):
	return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)

# 归一化操作
def norm(name, l_input, lsize=4):
	return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
    
   
def alex_ccnn10(_X, _weights, _biases, _dropout, w10, b10, o_layer="out"):
	global weights
	global biases
	lenx =36
	
	weights['wc1'] = _weights[26]
	weights['wc2'] = _weights[27]
	weights['wc3'] = _weights[28]
	weights['wc4'] = _weights[29]
	weights['wc5'] = _weights[30]
	weights['wc6'] = _weights[31]
	weights['wc7'] = _weights[32]
	weights['wc8'] = _weights[33]
	weights['wc9'] = _weights[34]
	
	biases['bc1'] = _biases[36]
	biases['bc2'] = _biases[37]
	biases['bc3'] = _biases[38]
	biases['bc4'] = _biases[39]
	biases['bc5'] = _biases[40]
	biases['bc6'] = _biases[41]
	biases['bc7'] = _biases[42]
	biases['bc8'] = _biases[43]
	biases['bc9'] = _biases[44]
	
	conv1 = conv2d('conv1', _X, weights['wc1'], biases['bc1'], 4, 'VALID')
	pool1 = max_pool('pool1', conv1, k=3,s=2)
	conv2 = conv2d('conv2', pool1, weights['wc2'], biases['bc2'], 1, 'SAME')
	pool2 = max_pool('pool2', conv2, k=3,s=2)
	conv3 = conv2d('conv3', pool2, weights['wc3'], biases['bc3'], 1, 'SAME')
	conv4 = conv2d('conv4', conv3, weights['wc4'], biases['bc4'], 1, 'SAME')
	conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'], 1, 'SAME')
	pool5 = max_pool('pool5', conv5, k=3,s=2)	
	conv6= conv2d('conv6', pool5, weights['wc6'], biases['bc6'], 1, 'SAME')
	fc7 = conv2d('fc7', conv6, weights['wc7'], biases['bc7'], 1, 'SAME')
	fc7 = tf.nn.dropout(fc7, _dropout)
	fca = conv2d('fca', fc7, weights['wc8'], biases['bc8'], 1, 'SAME')
	fca = tf.nn.dropout(fca, _dropout)
	fcb = conv2d('fcb', fca, weights['wc9'], biases['bc9'], 1, 'SAME')
	h = tf.reduce_max(fcb, reduction_indices=[1, 2]) # Global max Pooling

	out = tf.nn.softmax(tf.matmul(h, w10)+b10, name='fc3') # Relu activation
	if (o_layer=="out"):
		return out
	elif (o_layer=="smap"):
		return tf.reduce_max(fcb, reduction_indices=[3])
	elif (o_layer=="fcb"):
		return tf.reduce_max(fcb, reduction_indices=[1,2])
	elif (o_layer=="fca"):
		return tf.reduce_max(fca, reduction_indices=[1,2])
	else:
		eval('return tf.reduce_max(' + o_layer + ', reduction_indices=[1, 2])') # Global max Pooling
	


# 定义整个网络 
def alex_ccnn(_X, _weights, _biases, _dropout):
	global weights
	global biases
	lenx = 8

	weights['wc1'] = _weights[0]
	weights['wc2'] = _weights[1]
	weights['wc3'] = _weights[2]
	weights['wc4'] = _weights[3]
	weights['wc5'] = _weights[4]
	biases['bc1'] = _biases[8]
	biases['bc2'] = _biases[9]
	biases['bc3'] = _biases[10]
	biases['bc4'] = _biases[11]
	biases['bc5'] = _biases[12]
	
	conv1 = conv2d('conv1', _X, weights['wc1'], biases['bc1'], 4, 'VALID')
	pool1 = max_pool('pool1', conv1, k=3,s=2)
	conv2 = conv2d('conv2', pool1, weights['wc2'], biases['bc2'], 1, 'SAME')
	pool2 = max_pool('pool2', conv2, k=3,s=2)
	conv3 = conv2d('conv3', pool2, weights['wc3'], biases['bc3'], 1, 'SAME')
	conv4 = conv2d('conv4', conv3, weights['wc4'], biases['bc4'], 1, 'SAME')
	conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'], 1, 'SAME')
	pool5 = max_pool('pool5', conv5, k=3,s=2)	
	conv6= conv2d('conv6', pool5, weights['wc6'], biases['bc6'], 1, 'SAME')
	fc7 = conv2d('fc7', conv6, weights['wc7'], biases['bc7'], 1, 'SAME')
	fc7 = tf.nn.dropout(fc7, _dropout)
	fca = conv2d('fca', fc7, weights['wc8'], biases['bc8'], 1, 'SAME')
	fca = tf.nn.dropout(fca, _dropout)
	fcb = conv2d('fcb', fca, weights['wc9'], biases['bc9'], 1, 'SAME')
	h = tf.reduce_max(fcb, reduction_indices=[1, 2]) # Global max Pooling
	
	out = tf.nn.softmax(tf.matmul(h, weights['out']) + biases['out'], name='fc3') # Relu activation
	return out
	
def alex_ccnn2(_X,  _dropout, o_layer="out"):
	global weights
	global biases
	conv1 = conv2d('conv1', _X, weights['wc1'], biases['bc1'], 4, 'VALID')
	pool1 = max_pool('pool1', conv1, k=3,s=2)
	conv2 = conv2d('conv2', pool1, weights['wc2'], biases['bc2'], 1, 'SAME')
	pool2 = max_pool('pool2', conv2, k=3,s=2)
	conv3 = conv2d('conv3', pool2, weights['wc3'], biases['bc3'], 1, 'SAME')
	conv4 = conv2d('conv4', conv3, weights['wc4'], biases['bc4'], 1, 'SAME')
	conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'], 1, 'SAME')
	pool5 = max_pool('pool5', conv5, k=3,s=2)	
	conv6= conv2d('conv6', pool5, weights['wc6'], biases['bc6'], 1, 'SAME')
	fc7 = conv2d('fc7', conv6, weights['wc7'], biases['bc7'], 1, 'SAME')
	fc7 = tf.nn.dropout(fc7, _dropout)
	fca = conv2d('fca', fc7, weights['wc8'], biases['bc8'], 1, 'SAME')
	fca = tf.nn.dropout(fca, _dropout)
	fcb = conv2d('fcb', fca, weights['wc9'], biases['bc9'], 1, 'SAME')
	h = tf.reduce_max(fcb, reduction_indices=[1, 2]) # Global max Pooling

	out = tf.nn.softmax(tf.matmul(h, weights['out']) + biases['out'], name='fc3') # Relu activation
	if (o_layer=="out"):
		return out
	elif (o_layer=="smap"):
		return tf.reduce_max(fcb, reduction_indices=[3])
	elif (o_layer=="fcb"):
		return tf.reduce_max(fcb, reduction_indices=[1,2])
	else:
		eval('return tf.reduce_max(' + o_layer + ', reduction_indices=[1, 2])') # Global max Pooling

# 定义整个网络 
def alex_net(_X, _weights, _biases, _dropout):
	conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'], 4, 'VALID')
	pool1 = max_pool('pool1', conv1, k=3,s=2)
	conv2 = conv2d('conv2', pool1, _weights['wc2'], _biases['bc2'], 1, 'SAME')
	pool2 = max_pool('pool2', conv2, k=3,s=2)
	conv3 = conv2d('conv3', pool2, _weights['wc3'], _biases['bc3'], 1, 'SAME')
	conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'], 1, 'SAME')
	conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'], 1, 'SAME')
	pool5 = max_pool('pool5', conv5, k=3,s=2)	

	dense1 = tf.reshape(pool5, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
	dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation
	dd1=tf.nn.dropout(dense1, _dropout)
	dense2 = tf.nn.relu(tf.matmul(dd1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
	dd2=tf.nn.dropout(dense2, _dropout)
	out = tf.nn.softmax(tf.matmul(dd2, _weights['out']) + _biases['out'], name='fc3') # Relu activation
	return out



def getccnn(_X, _dropout, W, B):
	return alex_ccnn(_X, W, B, _dropout)


def getccnn2(_X, _dropout,  o_layer="fca"):
	return alex_ccnn2(_X, _dropout, o_layer=o_layer)
