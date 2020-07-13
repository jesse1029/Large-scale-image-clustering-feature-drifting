# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random as rn
from Queue import Queue
import threading
import time
import signal
import sys

from collections import namedtuple
from math import sqrt
import os

from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.examples.tutorials.mnist import input_data

def batch_norm(x, n_out, scope='bn'):
	
	with tf.device('/cpu:0'):
		mean, var = tf.nn.moments(x, axes=[0,1,2])
	beta = tf.Variable(tf.zeros([n_out]), name="beta")
	gamma = weight_variable([n_out], name="gamma")

	batch_norm = tf.nn.batch_norm_with_global_normalization(
		x, mean, var, beta, gamma, 0.001,
		scale_after_normalization=True)
	return batch_norm

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial, name=name)
    
# 卷积操作
def conv2d(l_input, w, b, s, p):
	n_out=w.get_shape().as_list()[3]
	return tf.nn.relu(batch_norm(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1,s,s,1], padding=p),b), n_out))

def res_net(x, activation=tf.nn.relu):
  """Builds a residual network.
  Note that if the input tensor is 2D, it must be square in order to be
  converted to a 4D tensor.
  Borrowed structure from:
  github.com/pkmital/tensorflow_tutorials/blob/master/10_residual_network.py
  Args:
    x: Input of the network
    y: Output of the network
    activation: Activation function to apply after each convolution
  Returns:
    Predictions and loss tensors.
  """

  # Configurations for each bottleneck group.
  BottleneckGroup = namedtuple(
      'BottleneckGroup', ['num_blocks', 'num_filters', 'bottleneck_size'])
  groups = [BottleneckGroup(3, 128, 32),
            BottleneckGroup(3, 256, 64),
            BottleneckGroup(3, 512, 128),
            BottleneckGroup(3, 1024, 256)]

  input_shape = x.get_shape().as_list()

  # Reshape the input into the right shape if it's 2D tensor
  if len(input_shape) == 2:
    ndim = int(sqrt(input_shape[1]))
    x = tf.reshape(x, [-1, ndim, ndim, 1])

  # First convolution expands to 64 channels
  with tf.variable_scope('conv_layer1'):
    net = conv2d(x, tf.Variable(tf.random_normal([7, 7, 3, 64], mean=0.0, stddev=0.01)), tf.Variable(tf.zeros([64])+0.1), 1, 'SAME')

  # Max pool
  net = tf.nn.max_pool(
      net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  # First chain of resnets
  with tf.variable_scope('conv_layer2'):
    net = conv2d(net, tf.Variable(tf.random_normal([3, 3, 64, 128], mean=0.0, stddev=0.01)), tf.Variable(tf.zeros([128])+0.1), 1, 'SAME')

  # Create the bottleneck groups, each of which contains `num_blocks`
  # bottleneck groups.
  for group_i, group in enumerate(groups):
    for block_i in range(group.num_blocks):
      name = 'group_%d/block_%d' % (group_i, block_i)

      # 1x1 convolution responsible for reducing dimension
      with tf.variable_scope(name + '/conv_in'):
        conv = conv2d(net, tf.Variable(tf.random_normal([3, 3, group.bottleneck_size, group.bottleneck_size], mean=0.0, stddev=0.01)), tf.Variable(tf.zeros([group.bottleneck_size])+0.1), 1, 'SAME')

      with tf.variable_scope(name + '/conv_bottleneck'):
        conv = conv2d(conv, tf.Variable(tf.random_normal([3, 3, group.bottleneck_size, group.bottleneck_size], mean=0.0, stddev=0.01)), tf.Variable(tf.zeros([group.bottleneck_size])+0.1), 1, 'SAME')
                                

      # 1x1 convolution responsible for restoring dimension
      with tf.variable_scope(name + '/conv_out'):
        input_dim = net.get_shape()[-1].value
        conv = conv2d(conv, tf.Variable(tf.random_normal([3, 3, input_dim, input_dim], mean=0.0, stddev=0.01)), tf.Variable(tf.zeros([input_dim])+0.1), 1, 'SAME')
        

      # shortcut connections that turn the network into its counterpart
      # residual function (identity shortcut)
      net = conv + net

    try:
      # upscale to the next group size
      next_group = groups[group_i + 1]
      with tf.variable_scope('block_%d/conv_upscale' % group_i):
        conv = conv2d(net, tf.Variable(tf.random_normal([3, 3, next_group.num_filters, next_group.num_filters], mean=0.0, stddev=0.01)), tf.Variable(tf.zeros([next_group.num_filters])+0.1), 1, 'SAME')

    except IndexError:
      pass

  net_shape = net.get_shape().as_list()
  net = tf.nn.avg_pool(net,
                       ksize=[1, net_shape[1], net_shape[2], 1],
                       strides=[1, 1, 1, 1], padding='VALID')

  net_shape = net.get_shape().as_list()
  net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])

  return net




batch_size = 64
training_iters = batch_size * 50000
display_step = 20

# 定义网络参数
n_input = (227,227,3) # 输入的维度
n_classes = 1000 # 标签的维度
dropout = 0.5 # Dropout 的概率

x = tf.placeholder(tf.float32, [None, 227,227,3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)


with tf.device('/cpu:0'):
	fn='/home/jess/Disk1/ilsvrc12/ILSVRC2012_img_train/'
	fn2='/home/jess/Disk1/ilsvrc12/label/train.txt'

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
				vbx  =vbx+1
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

load_thread = threading.Thread(target=load_data)
load_thread.deamon = True

def trainer():
# 构建模型
	pred = res_net(x)

	# 定义损失函数和学习步骤
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

	# 测试网络
	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# 初始化所有的共享变量
	init = tf.initialize_all_variables()
	gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.66))

	# 开启一个训练
	with tf.device('/cpu:0'):
		saver = tf.train.Saver()
	with tf.Session(config=gpu_options) as sess:
		sess.run(init)
		global step
		global data_queue
		step = 0
		lr = 1e-2
		#saver.restore(sess, "/Data/tf_alex_ccnn_model_iter27501.ckpt")
		#step = 27501
		# Keep training until reach max iterations
		while step * batch_size < training_iters:
			epcho1 = np.floor((step*batch_size)/glen1)
			if (((step*batch_size)%glen1 < batch_size) & (epcho1>0)):
				lr /= 10

			batch_xs, batch_ys = data_queue.get()
			# 获取批数据
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout, learning_rate: lr})
			if (step % 2500==1):
				with tf.device('/cpu:0'):
					save_path = saver.save(sess, "Data/tf_alex_ccnn_model_iter" + str(step) + ".ckpt", global_step=step)
					print("Model saved in file at iteration %d: %s" % (step*batch_size,save_path))
				
			if step % display_step == 1:
				
				# 计算损失值
				loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
				acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
				
				print("Iter=" + str(step*batch_size) + "/epcho=" + str(np.floor((step*batch_size)/glen1)) + ", Loss= " + "{:.5f}".format(loss) + ", Training Accuracy=" + "{:.5f}".format(acc))
			step += 1
			
			
		print("Optimization Finished!")
		#saver.save(sess, 'jigsaw', global_step=step)
		save_path = saver.save(sess, "Data/tf_alex_ccnn_model.ckpt", global_step=step)
		print("Model saved in file: %s" % save_path)
		# 计算测试精度
		batch_xs, batch_ys = imagenet_batch(4096, 55)
		print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.}))
	threading.Thread._Thread__stop(load_thread)

def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        global load_thread
        
        threading.Thread._Thread__stop(load_thread)
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


if __name__ == '__main__':
	with tf.device('/gpu:0'):
		trainer()
