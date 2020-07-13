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
import models
import lenetccnnmodel as al
import signal
import sys
from sklearn.metrics.cluster import normalized_mutual_info_score as mi
from sklearn.cluster import AffinityPropagation
import gc
from mnist import MNIST
import gzip 
import cPickle
import numpy as np
mndata = MNIST('../DCN/python-mnist/data')
train_x, train_y=mndata.load_training()
train_x=np.array(train_x)
train_y=np.array(train_y)
print train_x.shape
print train_y.shape


# 定义网络超参数


batch_size = 50
training_iters = batch_size*4 * 200000
display_step = 200
n_classes = 10# 标签的维度
dropout = 0.5 # Dropout 的概率
# 定义网络参数
n_input = (224,224,3) # 输入的维度
train_x=np.reshape(train_x,(60000,28,28))



x = tf.placeholder(tf.float32, [None, 28,28,1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
lr = 0.1
learning_rate = tf.placeholder(tf.float32)

vlen1 = len(train_x)
seq1 = range(vlen1)
rn.shuffle(seq1)
batch_num = 0

def val_batch(bs, ss=-1):
	global batch_num
	size = 28, 28
	len1 = len(train_x)
	data = np.zeros((bs,28,28,1))
	label1 = np.zeros((bs, n_classes));
	st = bs*batch_num if ss==-1 else bs*ss
	cnt = 0
	vbx = 0
	global seq1
	
	global phase
	for k in range(bs):
		idx1 = k+st
		if (idx1 >= len1):
			idx1 = idx1 % len1
		idx1 = seq1[idx1]
		im = train_x[idx1,:,:]
		im = np.resize(im, (28,28,1)) 
		label1[cnt, train_y[idx1]] = 1
		data[cnt,:,:]=im
		cnt = cnt + 1
	batch_num=batch_num+1
	return data, label1

	

#async load data
with tf.device('/cpu:0'):
	data_queue = Queue(100)

	def load_data():
		while True:
			if Queue.qsize(data_queue)<100:
				#print 'loading next batch...'+str(Queue.qsize(data_queue))
				x,y = val_batch(batch_size)
				data_queue.put((x,y))
	load_thread = threading.Thread(target=load_data)
	load_thread.deamon=True

def trainer():
	global load_thread
	global seq1
	global step
	global data_queue
	global lr
	X = tf.placeholder("float", [None, 28, 28,1])
	sX = tf.placeholder("float", [None, 7,7])
	Y = tf.placeholder("float", [None, n_classes])	
	X2 = tf.placeholder(tf.float32,  [None, 28, 28,1])
	
	# 构建模型

	ttk = time.time()
	with tf.Session() as sess:
		

		pred = al.lenet_ccnn(X, None, None, 0.5)
		fcb = al.lenet_ccnn_fcb(X, None, None, 0.5)
		cost = -tf.reduce_sum(Y*tf.log(pred))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		
		#==========Get Init Center====================@
		#~ centers, lab1=val_batch(n_classes, ss=0)
		#~ rn.shuffle(seq1)
		vlen2=vlen1+(batch_size-(vlen1%batch_size))
		saver = tf.train.Saver()		
		gtlab = np.zeros((vlen2))
		predicts = np.zeros((vlen2))
		weightsInd = 2
		biasesInd = 6
		lr = 5e-6
		step = 1
		
		sess.run(tf.initialize_all_variables())
		
		allRuntime=0
		load_thread = threading.Thread(target=load_data)
		load_thread.deamon=True
		load_thread.start()	
		
		for v in range(100):
			ts=time.time()

			rn.shuffle(seq1)
			bct = 0
			
			for k in range(0, vlen2, batch_size):
				epcho1=np.floor((step*batch_size)/vlen1)
				batch_sx, batch_ys = data_queue.get()				
					#===========Update network====================
				sess.run(optimizer, feed_dict={X:batch_sx, Y:batch_ys, keep_prob:dropout, learning_rate:lr})
					
				if step % display_step == 1:
					# 计算损失值
					loss = sess.run(cost, feed_dict={X: batch_sx, Y: batch_ys, keep_prob: 1.})
					acc = sess.run(accuracy, feed_dict={X: batch_sx, Y: batch_ys, keep_prob: 1.})
					print "Iter=" + str(step*batch_size) + "/epcho=" + str(v) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy=" + "{:.5f}".format(acc) + ", lr=" + str(lr) + ", runtime=" + str(time.time()-ttk)
					ttk = time.time()
				
				step += 1


		td = time.time()-ts
		allRuntime=allRuntime+td
		save_path = saver.save(sess, "/Data/tf_mnist_s.ckpt")
		print("Model saved in file: %s" % save_path)
		threading.Thread._Thread__stop(load_thread)
			

	threading.Thread._Thread__stop(load_thread)

def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        global load_thread
        
        threading.Thread._Thread__stop(load_thread)
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
	trainer()
