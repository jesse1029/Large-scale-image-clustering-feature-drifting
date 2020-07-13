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

import signal
import sys

# 定义网络超参数


batch_size = 8
training_iters = batch_size * 400000
display_step = 20

# 定义网络参数
n_input = (227,227,3) # 输入的维度
n_classes = 1000 # 标签的维度
dropout = 0.5 # Dropout 的概率

keep_prob = tf.placeholder(tf.float32)
lr = 0.1
learning_rate = tf.placeholder(tf.float32)


fn='/Data/jess/ILSVRC2012_img_train/'
fn2='/home/jess/caffe-old/data/ilsvrc12/train.txt'
fnv='/Data/jess/ILSVRC2012_img_val/'
fn2v='/home/jess/caffe-old/data/ilsvrc12/val.txt'
with open(fn2) as fx:
	content = fx.readlines()
with open(fn2v) as fx:
	contentval = fx.readlines()
glen1 = len(content)
seq1 = range(0,glen1)
rn.shuffle(seq1)

glen1v = len(contentval)
seq1v = range(0,glen1v)
rn.shuffle(seq1v)

batch_num = 0
phase = "TRAIN"
def imagenet_batch(bs):
	global batch_num
	global phase
	size = 256, 256
	len1 = len(contentval) if phase=="VAL" else len(content)
	bs = bs*4  if phase=="VAL" else bs
	mul = 1 if phase=="VAL" else 4
	data = np.zeros((bs*mul,227,227,3))
	label1 = np.zeros((bs*mul, n_classes));
	st = bs*batch_num
	#print("batch size="+str(bs)+",step="+str(batch_num))	
	cnt = 0
	vbx = 0
	global seq1
	global seq1v
	seq = seq1v if phase=="VAL" else seq1
	cont = contentval if phase=="VAL" else content
	
	mean1= (104.0 ,117.0 ,123.0)
	
	for k in range(bs):
		idx1 = k+st
		if (idx1 >= len1):
			idx1 = idx1 % len1
		
		idx1 = seq[idx1]
		fn1 = cont[idx1].replace('\n','')
		fn2 = fn1.split(' ')
		
		fnx = fn+fn2[0] if phase=="TRAIN" else fnv+fn2[0]
		fnx = fnx.replace(" ","")
		
		try:
			im = cv2.imread(fnx)
			im = np.asarray(im)
			im = cv2.resize(im, (256,256)) - mean1

			data[cnt,:,:,:]=cropping(im)
			if phase=="TRAIN":
				data[cnt+1,:,:,:]=cv2.flip(cropping(im), 1, dst=None)
				data[cnt+2,:,:,:]=cv2.flip(cropping(im), 0, dst=None)
				data[cnt+3,:,:,:]=cv2.flip(cropping(im), -1, dst=None)	
				label1[range(cnt,cnt+4), int(fn2[1])] = 1
				cnt = cnt + 4
			else:
				label1[cnt, int(fn2[1])] = 1
				cnt = cnt + 1
				
		except Exception:
			pass

	if vbx>0:
		print("There are " + str(vbx) + " times that exception is happen.")	
	batch_num = batch_num + 1
	return data, label1

def cropping(im):
	global phase
	row,col,channels = im.shape
	rst = 14
	cst = 14
	
	if (phase=="TRAIN"):
		rst = rn.randint(0,14)
		cst = rn.randint(0,14)
	im = im[rst:rst+227, cst:cst+227,:]
	return im

#async load data

data_queue = Queue(32)

def load_data():
	while True:
		if Queue.qsize(data_queue)<32:
			#print 'loading next batch...'+str(Queue.qsize(data_queue))
			x,y = imagenet_batch(batch_size)
			data_queue.put((x,y))
		time.sleep(0.001)

load_thread = threading.Thread(target=load_data)
load_thread.deamon=True

def trainer():
	global phase
	phase="TRAIN"
	global load_thread
	load_thread.start()
	X = tf.placeholder("float", [None, 227, 227, 3])
	Y = tf.placeholder("float", [None, n_classes])
	
	# ResNet Models
	pred = models.resnet(X, 56)
	#net = models.resnet(X, 32)
	# net = models.resnet(X, 44)
	# net = models.resnet(X, 56)

	cost = -tf.reduce_sum(Y*tf.log(pred))
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	info2=tf.argmax(pred,1)

	# 初始化所有的共享变量
	init = tf.initialize_all_variables()
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.)

	# 开启一个训练
	
	saver = tf.train.Saver()
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run(init)
		global step
		global data_queue
		global lr
		
		global load_thread
		lr = 1e-4
		step = 0
		saver.restore(sess, "Data/tf_resnet_ccnn_model_iter65001.ckpt-65001")
		step = 65002
		# Keep training until reach max iterations
		
		while step * batch_size < training_iters:
			epcho1=np.floor((step*batch_size)/glen1)
			if (((step*batch_size)%glen1 < batch_size) & (epcho1>0)):
				lr /= 10

			batch_xs, batch_ys = data_queue.get()
			# 获取批数据
			sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: dropout, learning_rate: lr})
			if (step % 2500==1) & (step>1):
				threading.Thread._Thread__stop(load_thread)
				with data_queue.mutex:
					data_queue.queue.clear()
				phase="VAL"
				load_thread = threading.Thread(target=load_data)
				load_thread.start()
				time.sleep(1)
				acc2 = 0.0
				for z in range(glen1v/(batch_size*4)):
					batch_xs, batch_ys = data_queue.get()
					acc2 = acc2+sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.})
				acc2 = acc2 / (glen1v/(batch_size*4))
				save_path = saver.save(sess, "Data/tf_resnet_ccnn_model_iter" + str(step) + ".ckpt", global_step=step)
				print("Model saved in file at iteration %d: %s" % (step*batch_size,save_path))
				print "Iter=" + str(step*batch_size) + "/epcho=" + str(np.floor((step*batch_size)/glen1)) + ", Validation Accuracy=" + "{:.5f}".format(acc2) + ", lr=" + str(lr)
				threading.Thread._Thread__stop(load_thread)
				with data_queue.mutex:
					data_queue.queue.clear()
				phase="TRAIN"
				load_thread = threading.Thread(target=load_data)
				load_thread.start()
				time.sleep(1)
				
			if step % display_step == 1:
				
				# 计算损失值
				loss = sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.})
				acc = sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.})
				#info=sess.run(info2,feed_dict={X:batch_xs, keep_prob:1.})
				#print(info)
				print "Iter=" + str(step*batch_size) + "/epcho=" + str(np.floor((step*batch_size)/glen1)) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy=" + "{:.5f}".format(acc) + ", lr=" + str(lr)
			step += 1
			
			
		print "Optimization Finished!"
		#saver.save(sess, 'jigsaw', global_step=step)
		save_path = saver.save(sess, "Data/tf_resnet_ccnn_model.ckpt", global_step=step)
		print("Model saved in file: %s" % save_path)
		
	threading.Thread._Thread__stop(load_thread)

def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        global load_thread
        
        threading.Thread._Thread__stop(load_thread)
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
	
	trainer()

