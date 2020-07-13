# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

n_classes = 1000 # 标签的维度
dropout = 0.5 # Dropout 的概率

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
    
# 定义整个网络 
def alex_ccnn(_X, _weights, _biases, _dropout):
	conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'], 4, 'VALID')
	pool1 = max_pool('pool1', conv1, k=3,s=2)
	conv2 = conv2d('conv2', pool1, _weights['wc2'], _biases['bc2'], 1, 'SAME')
	pool2 = max_pool('pool2', conv2, k=3,s=2)
	conv3 = conv2d('conv3', pool2, _weights['wc3'], _biases['bc3'], 1, 'SAME')
	conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'], 1, 'SAME')
	conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'], 1, 'SAME')
	pool5 = max_pool('pool5', conv5, k=3,s=2)	
	conv6 = conv2d('conv6', pool5, _weights['wc6'], _biases['bc6'], 1, 'SAME')
	
	fc7 = conv2d('fc7', conv6, _weights['wc7'], _biases['bc7'], 1, 'SAME')
	fc7 = tf.nn.dropout(fc7, _dropout)
	fca = conv2d('fca', fc7, _weights['wc8'], _biases['bc8'], 1, 'SAME')
	fca = tf.nn.dropout(fca, _dropout)
	fcb = conv2d('fcb', fca, _weights['wc9'], _biases['bc9'], 1, 'SAME')
	h = tf.reduce_max(fcb, reduction_indices=[1, 2]) # Global max Pooling

	return h

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

def load_with_skip(data_path, session, skip_layer):
	data_dict = np.load(data_path).item()
	for key in data_dict:
		if key not in skip_layer:
			with tf.variable_scope(key, reuse=True):
				for subkey, data in zip(('weights', 'biases'), data_dict[key]):
					session.run(tf.get_variable(subkey).assign(data))
					

# 存储所有的网络参数
with tf.device('/cpu:0'):
	weights = {
		'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=0.01)),
		'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=0.01)),
		'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=0.01)),
		'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384], stddev=0.01)),
		'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=0.01)),
		'wd1': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=0.01)),
		'wd2': tf.Variable(tf.random_normal([4096, 4096], stddev=0.01)),
		'out': tf.Variable(tf.random_normal([4096, n_classes], stddev=0.01))
	}
	biases = {
		'bc1': tf.Variable(tf.random_normal([96], mean=0.1, stddev=0.01)),
		'bc2': tf.Variable(tf.random_normal([256], mean=0.1, stddev=0.01)),
		'bc3': tf.Variable(tf.random_normal([384], mean=0.1, stddev=0.01)),
		'bc4': tf.Variable(tf.random_normal([384], mean=0.1, stddev=0.01)),
		'bc5': tf.Variable(tf.random_normal([256], mean=0.1, stddev=0.01)),
		'bd1': tf.Variable(tf.random_normal([4096], mean=0.1, stddev=0.01)),
		'bd2': tf.Variable(tf.random_normal([4096], mean=0.1, stddev=0.01)),
		'out': tf.Variable(tf.random_normal([n_classes], mean=0.1, stddev=0.01))
	}
	
	#~ weights2 = {
		#~ 'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=0.01)),
		#~ 'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=0.01)),
		#~ 'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=0.01)),
		#~ 'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384], stddev=0.01)),
		#~ 'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=0.01)),
		#~ 'wc6': tf.Variable(tf.random_normal([3, 3, 256, 6144], stddev=0.01)),
		#~ 'wc7': tf.Variable(tf.random_normal([1, 1, 6144, 6144], stddev=0.01)),
		#~ 'wc8': tf.Variable(tf.random_normal([3, 3, 6144, 2048], stddev=0.01)),
		#~ 'wc9': tf.Variable(tf.random_normal([3, 3, 2048, n_classes], stddev=0.01)),
		#~ 'out': tf.Variable(tf.random_normal([4096, n_classes], stddev=0.01))
	#~ }
	#~ biases2 = {
		#~ 'bc1': tf.Variable(tf.random_normal([96], mean=0.1, stddev=0.01)),
		#~ 'bc2': tf.Variable(tf.random_normal([256], mean=0.1, stddev=0.01)),
		#~ 'bc3': tf.Variable(tf.random_normal([384], mean=0.1, stddev=0.01)),
		#~ 'bc4': tf.Variable(tf.random_normal([384], mean=0.1, stddev=0.01)),
		#~ 'bc5': tf.Variable(tf.random_normal([256], mean=0.1, stddev=0.01)),
		#~ 'bc6': tf.Variable(tf.random_normal([6144], mean=0.1, stddev=0.01)),
		#~ 'bc7': tf.Variable(tf.random_normal([6144], mean=0.1, stddev=0.01)),
		#~ 'bc8': tf.Variable(tf.random_normal([2048], mean=0.1, stddev=0.01)),
		#~ 'bc9': tf.Variable(tf.random_normal([n_classes], mean=0.1, stddev=0.01)),
		#~ 'out': tf.Variable(tf.random_normal([n_classes], mean=0.1, stddev=0.01))
	#~ }

def getalex(_X, _dropout):
	return alex_net(_X, weights, biases, _dropout)

#~ def getccnn(_X, _dropout):
	#~ return alex_ccnn(_X, weights2, biases2, _dropout)
