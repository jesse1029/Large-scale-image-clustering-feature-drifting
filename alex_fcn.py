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
import alexmodel as al
import signal
import sys

# 定义网络超参数


batch_size = 32
training_iters = batch_size*4 * 200000
display_step = 20
n_classes = 1000 # 标签的维度
dropout = 0.5 # Dropout 的概率
# 定义网络参数
n_input = (224,224,3) # 输入的维度


x = tf.placeholder(tf.float32, [None, 224,224,3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
lr = 0.1
learning_rate = tf.placeholder(tf.float32)


fn='/home/jess/Disk1/ilsvrc12/ILSVRC2012_img_train/'
fn2='/home/jess/Disk1/ilsvrc12/label/train.txt'

fnv='/home/jess/Disk1/ilsvrc12/ILSVRC2012_img_val/'
fnv2='/home/jess/Disk1/ilsvrc12/label/val.txt'

with open(fn2) as fx:
	content = fx.readlines()
with open(fnv2) as fx2:
	valcontent = fx2.readlines()
	
glen1 = len(content)
vlen1 = len(valcontent)

seq1 = range(0,glen1)

rn.shuffle(seq1)
batch_num = 0
phase = "TRAIN"
def imagenet_batch(bs):
	global batch_num
	size = 256, 256
	len1 = len(content)
	data = np.zeros((bs*4,224,224,3))
	label1 = np.zeros((bs*4, n_classes));
	st = bs*batch_num
	#print("batch size="+str(bs)+",step="+str(batch_num))	
	cnt = 0
	vbx = 0
	global seq1
	mean1= (104.0 ,117.0 ,123.0)
	global phase
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

			
			label1[range(cnt,cnt+4), int(fn2[1])] = 1
			data[cnt,:,:,:]=cropping(im)
			data[cnt+1,:,:,:]=cv2.flip(cropping(im), 1, dst=None)
			data[cnt+2,:,:,:]=cv2.flip(cropping(im), 0, dst=None)
			data[cnt+3,:,:,:]=cv2.flip(cropping(im), -1, dst=None)
			
			cnt = cnt + 4
		except Exception:
			pass

	if vbx>0:
		print("There are " + str(vbx) + " times that exception is happen.")	
	batch_num = batch_num + 1
	return data, label1

def val_batch(bs, size1):
	size = 256, 256
	len1 = len(valcontent)
	data = np.zeros((bs,224,224,3))
	label1 = np.zeros((bs, n_classes));
	st = bs*size1
	cnt = 0
	vbx = 0
	global seq1
	mean1= (104.0 ,117.0 ,123.0)
	global phase
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
			label1[cnt, int(fn2[1])] = 1
			data[cnt,:,:,:]=cropping(im)
			cnt = cnt + 1
		except Exception:
			pass

	if vbx>0:
		print("There are " + str(vbx) + " times that exception is happen.")	

	return data, label1

def cropping(im, phase="TRAIN"):
	row,col,channels = im.shape
	rst = 16
	cst = 16
	
	if (phase=="TRAIN"):
		rst = rn.randint(0,16)
		cst = rn.randint(0,16)
	im = im[rst:rst+224, cst:cst+224,:]
	return im
	

#async load data

with tf.device('/cpu:0'):
	data_queue = Queue(64)

	def load_data():
		while True:
			if Queue.qsize(data_queue)<64:
				#print 'loading next batch...'+str(Queue.qsize(data_queue))
				x,y = imagenet_batch(batch_size)
				data_queue.put((x,y))
			time.sleep(0.05)

	load_thread = threading.Thread(target=load_data)
	load_thread.deamon=True
def get_weights():
  return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv5') if v.name.endswith('weights:0')]

def trainer():
	global load_thread
	load_thread.start()
	X = tf.placeholder("float", [None, 224, 224, 3])
	Y = tf.placeholder("float", [None, n_classes])
	# 构建模型
	oldpred = al.getalex(X, keep_prob)
	sess2 = tf.Session()
	sess2.run(tf.initialize_all_variables())
	saver = tf.train.Saver()
	saver.restore(sess2, "/home/jess/Disk1/ilsvrc12/tf_resnet_ccnn_model_iter252501.ckpt-252501")
	xx =tf.trainable_variables()
	sess2.close()
	

	import alexccnnmodel as al2
	pred = al2.getccnn(X, keep_prob, xx, xx)
	cost = -tf.reduce_sum(Y*tf.log(pred))
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	info2=tf.argmax(pred,1)
	# 初始化所有的共享变量
	init = tf.initialize_all_variables()
	# 开启一个训练
	#conv1 = al.conv2d('conv11', x, al.weights['wc1'], al.biases['bc1'], 4, 'VALID')
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		global step
		global data_queue
		global lr
		lr = 1e-3
		step = 0
		
		# Keep training until reach max iterations
		
		while step * batch_size < training_iters:
			epcho1=np.floor((step*batch_size)/glen1)
			if (((step*batch_size)%glen1 < batch_size) & (epcho1>0) & (epcho1 % 10==0)):
				lr /= 10

			batch_xs, batch_ys = data_queue.get()
			# 获取批数据
			sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: dropout, learning_rate: lr})
			if (step % 2500==1) & (step>2500):
				save_path = saver.save(sess, "/home/jess/Disk1/ilsvrc12/tf_alex_pre_ccnn_model_iter" + str(step) + ".ckpt", global_step=step)
				print("Model saved in file at iteration %d: %s" % (step*batch_size,save_path))
				acc2 = 0.0
				for kk in range(0, vlen1 / batch_size):
					batch_xs2, batch_ys2=val_batch(batch_size, kk)
					acc = sess.run(accuracy, feed_dict={X: batch_xs2, Y: batch_ys2, keep_prob: 1.})
					acc2 = acc2 + acc
				print "Validation accuracy: " + str(acc2/(vlen1/batch_size))
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
		save_path = saver.save(sess, "/home/jess/Disk1/ilsvrc12/tf_alex_pre_ccnn_model.ckpt", global_step=step)
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

