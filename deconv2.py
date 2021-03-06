# -*- coding: utf-8 -*-
# 输入数据

import os
import cv2
import numpy as np
import random as rn
import tensorflow as tf
from Queue import Queue
import threading
import time
# 定义网络超参数


learning_rate = 0.01

batch_size = 128
training_iters = batch_size * 200000
display_step = 20

# 定义网络参数
n_input = (227,227,3) # 输入的维度
n_classes = 1000 # 标签的维度
dropout = 0.5 # Dropout 的概率

x = tf.placeholder(tf.float32, [None, 227,227,3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)



fn='/Data/jess/ILSVRC2012_img_train/'
fn2='/home/jess/caffe-old/data/ilsvrc12/train.txt'

with open(fn2) as fx:
	content = fx.readlines()
glen1 = len(content)
seq1 = range(0,glen1)
rn.shuffle(seq1)
batch_num = 0
def imagenet_batch(bs):
	global batch_num
	size = 256, 256
	len1 = len(content)
	data = np.zeros((bs,227,227,3))
	label1 = np.zeros((bs, n_classes));
	st = bs*batch_num
	#print("batch size="+str(bs)+",step="+str(batch_num))	
	cnt = 0
	vbx = 0
	global seq1
	mean1= (104 ,117 ,123)
	for k in range(bs):
		idx1 = k+st
		if (idx1 >= len1):
			idx1 = idx1 % len1
		
		idx1 = seq1[idx1]
		fn1 = content[idx1].replace('\n','')
		fn2 = fn1.split(' ')
		
		fnx = fn+fn2[0]
		fnx = fnx.replace(" ","")
		
		try:
			im = cv2.imread(fnx)
			im = np.asarray(im)
			im = cv2.resize(im, (256,256)) - mean1
			im = im[14:227+14, 14:227+14,:]
			row,col,channels = im.shape

			label1[cnt, int(fn2[1])] = 1
			data[cnt,:,:,:]=im
			cnt = cnt + 1
		except Exception:
			pass

	
	
	if vbx>0:
		print("There are " + str(vbx) + " times that exception is happen.")	
	batch_num = batch_num + 1
	return data, label1
	
#async load data

data_queue = Queue(8)

def load_data():
	while True:
		if Queue.qsize(data_queue)<8:
			#print 'loading next batch...'+str(Queue.qsize(data_queue))
			x,y = imagenet_batch(batch_size)
			data_queue.put((x,y))
		time.sleep(1)




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

import signal
import sys

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
def alex_net(_X, _weights, _biases, _dropout):
	# 向量转为矩阵
	#_X = tf.reshape(_X, shape=[-1, 28, 28, 1])
	#_X = tf.reshape(_X, shape=[-1, 256, 256, 3])

	# 卷积层
	conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'], 4, 'VALID')
	conv1 = tf.nn.dropout(conv1, _dropout)
	# 下采样层
	
	pool1 = max_pool('pool1', conv1, k=3,s=2)
	# 卷积
	conv2 = conv2d('conv2', pool1, _weights['wc2'], _biases['bc2'], 1, 'SAME')
	conv2 = tf.nn.dropout(conv2, _dropout)

	# 下采样
	pool2 = max_pool('pool2', conv2, k=3,s=2)
		
	# 卷积
	conv3 = conv2d('conv3', pool2, _weights['wc3'], _biases['bc3'], 1, 'SAME')
	#conv4
	conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'], 1, 'SAME')
	#conv5
	conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'], 1, 'SAME')
	conv5 = tf.nn.dropout(conv5, _dropout)
	# 下采样
	pool5 = max_pool('pool5', conv5, k=3,s=2)	
	
	conv6 = conv2d('conv6', pool5, _weights['wc6'], _biases['bc6'], 1, 'SAME')
	
	fc7 = conv2d('fc7', conv6, _weights['wc7'], _biases['bc7'], 1, 'SAME')
	fc7 = tf.nn.dropout(fc7, _dropout)
	fca = conv2d('fca', fc7, _weights['wc8'], _biases['bc8'], 1, 'SAME')
	fca = tf.nn.dropout(fca, _dropout)
	fcb = conv2d('fcb', fca, _weights['wc9'], _biases['bc9'], 1, 'SAME')

	h = tf.reduce_max(fcb, reduction_indices=[1, 2]) # Global max Pooling
	out = tf.nn.relu(tf.matmul(h, _weights['out']) + _biases['out'], name='out')
	
	# Fully connected layer
	#dense1 = tf.reshape(pool5, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
	#dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation

	#dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

	# Output, class prediction
	#out = tf.matmul(dense2, _weights['out']) + _biases['out']
	#dense3 = tf.nn.relu(tf.matmul(dense2, weights['out']) + biases['out'], name='fc3') # Relu activation
	return out

def trainer():
	load_thread = threading.Thread(target=load_data)
	load_thread.start()
	# 存储所有的网络参数
	weights = {
		'wc1': tf.Variable(tf.random_normal([7, 7, 3, 96])),
		'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
		'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
		'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
		'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
		'wc6': tf.Variable(tf.random_normal([3, 3, 256, 6144])),
		'wc7': tf.Variable(tf.random_normal([1, 1, 6144, 6144])),
		'wc8': tf.Variable(tf.random_normal([3, 3, 6144, 2048])),
		'wc9': tf.Variable(tf.random_normal([3, 3, 2048, n_classes])),
		'out': tf.Variable(tf.random_normal([n_classes, n_classes]))
	}
	biases = {
		'bc1': tf.Variable(tf.random_normal([96])),
		'bc2': tf.Variable(tf.random_normal([256])),
		'bc3': tf.Variable(tf.random_normal([384])),
		'bc4': tf.Variable(tf.random_normal([384])),
		'bc5': tf.Variable(tf.random_normal([256])),
		'bc6': tf.Variable(tf.random_normal([6144])),
		'bc7': tf.Variable(tf.random_normal([6144])),
		'bc8': tf.Variable(tf.random_normal([2048])),
		'bc9': tf.Variable(tf.random_normal([n_classes])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}

	# 构建模型
	pred = alex_net(x, weights, biases, keep_prob)

	# 定义损失函数和学习步骤
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

	# 测试网络
	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# 初始化所有的共享变量
	init = tf.initialize_all_variables()
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

	# 开启一个训练
	
	saver = tf.train.Saver()
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run(init)
		global step
		global data_queue
		step = 0
		#saver.restore(sess, "/Data/tf_alex_ccnn_model_iter27501.ckpt")
		#step = 27501
		# Keep training until reach max iterations
		while step * batch_size < training_iters:
			#batch_xs, batch_ys = imagenet_batch(batch_size, step-1)

			batch_xs, batch_ys = data_queue.get()
			# 获取批数据
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
			if (step % 2500==1):

				save_path = saver.save(sess, "Data/tf_alex_ccnn_model_iter" + str(step) + ".ckpt", global_step=step)
				print("Model saved in file at iteration %d: %s" % (step*batch_size,save_path))
					
			if step % display_step == 1:
				
				# 计算损失值
				loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
				acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
				
				print "Iter=" + str(step*batch_size) + "/epcho=" + str(np.floor((step*batch_size)/glen1)) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy=" + "{:.5f}".format(acc)
			step += 1
			
			
		print "Optimization Finished!"
		#saver.save(sess, 'jigsaw', global_step=step)
		save_path = saver.save(sess, "Data/tf_alex_ccnn_model.ckpt", global_step=step)
		print("Model saved in file: %s" % save_path)
		# 计算测试精度
		#batch_xs, batch_ys = imagenet_batch(4096, 0)
		#print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
		load_thread.stop()
		load_thread.join()

if __name__ == '__main__':
	
	trainer()

