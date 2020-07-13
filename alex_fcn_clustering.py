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
from sklearn.metrics.cluster import normalized_mutual_info_score as mi
from sklearn.cluster import AffinityPropagation
import gc

# 定义网络超参数


batch_size = 50
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



fnv='Data/ILSVRC2012_img_val/'
fnv2='Data/label/val.txt'

with open(fnv2) as fx2:
	valcontent = fx2.readlines()
	
vlen1 = len(valcontent)
seq1 = range(vlen1)
rn.shuffle(seq1)
batch_num = 0

def val_batch(bs, ss=-1):
	global batch_num
	size = 256, 256
	len1 = len(valcontent)
	data = np.zeros((bs,224,224,3))
	label1 = np.zeros((bs, n_classes));
	st = bs*batch_num if ss==-1 else bs*ss
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
		fn1 = valcontent[idx1].replace('\n','')
		fn2 = fn1.split(' ')
		fnx = fnv+fn2[0]
		fnx = fnx.replace(" ","")
		im = cv2.imread(fnx)
		im = np.asarray(im)
		im = cv2.resize(im, (256,256)) - mean1
		label1[cnt, int(fn2[1])] = 1
		data[cnt,:,:,:]=cropping(im)
		cnt = cnt + 1
	batch_num=batch_num+1
	return data, label1

def cropping(im, phase="TRAIN"):
	row,col,channels = im.shape
	rst = 16
	cst = 16
	im = im[rst:rst+224, cst:cst+224,:]
	return im
	
X2= tf.placeholder(tf.float32,  [None, 224, 224, 3])
X = tf.placeholder(tf.float32, [None, 224, 224, 3])
Y = tf.placeholder(tf.float32, [None, n_classes])
def getFeat(data, isinit, o_layer="out"):
	global n_classes, xx, sess2, saver, X, X2, Y
	if (isinit==True):
		oldpred = al.getalex(X, keep_prob)
		sess2 = tf.Session()
		sess2.run(tf.initialize_all_variables())
		saver = tf.train.Saver()
		saver.restore(sess2, "Data/tf_pretrained.ckpt")
		xx =tf.trainable_variables()
		sess2.close()
	global sess, pred
	if (isinit==True):
		import alexccnnmodel as al2
		pred = al2.getccnn2(X2, keep_prob,  o_layer=o_layer)
		init = tf.initialize_all_variables()
		saver = tf.train.Saver()
		sess=tf.Session() 
		sess.run(init)
		saver.restore(sess, "Data/tf_alex_pre_ccnn_model_iter285001.ckpt-285001")
	return sess.run(pred, feed_dict={X2:data, keep_prob:1.})

#async load data

with tf.device('/cpu:0'):
	data_queue = Queue(50)

	def load_data():
		while True:
			if Queue.qsize(data_queue)<50:
				#print 'loading next batch...'+str(Queue.qsize(data_queue))
				x,y = val_batch(batch_size)
				data_queue.put((x,y))
			time.sleep(0.05)

	load_thread = threading.Thread(target=load_data)
	load_thread.deamon=True
def get_weights():
  return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv5') if v.name.endswith('weights:0')]



def trainer():
	global load_thread
	X = tf.placeholder("float", [None, 224, 224, 3])
	sX = tf.placeholder("float", [None, 7,7])
	Y = tf.placeholder("float", [None, n_classes])	
	X2 = tf.placeholder(tf.float32,  [None, 224, 224, 3])

	# 构建模型
	oldpred = al.getalex(X, keep_prob)
	sess2 = tf.Session()
	sess2.run(tf.initialize_all_variables())
	saver = tf.train.Saver()
	saver.restore(sess2, "Data/tf_pretrained.ckpt")
	xx =tf.trainable_variables()
	sess2.close()
	
	import alexccnnmodel as al2
	pred = al2.getccnn(X, keep_prob, xx, xx)
	#smap = al2.getccnn2(X, keep_prob, o_layer='smap')
	#cost2 = tf.reduce_mean(tf.abs(smap-sX), [1, 2])
	cost = -tf.reduce_sum(Y*tf.log(pred)) #+ cost2
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	cp = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	acc2 = tf.reduce_mean(tf.cast(cp, "float"))
	info2=tf.argmax(pred,1)
	# 初始化所有的共享变量
	init = tf.initialize_all_variables()

	#==========Get Init Center====================@
	global seq1
	#~ centers, lab1=val_batch(n_classes, ss=0)
	#~ rn.shuffle(seq1)
	saver = tf.train.Saver()
	
	load_thread.start()
	gtlab = np.zeros((vlen1))
	predicts = np.zeros((vlen1))

	feaStr = 'out'
	feed_x = tf.placeholder("float", al2.biases[feaStr].get_shape().as_list())
	feed_y = tf.placeholder("float", al2.weights[feaStr].get_shape().as_list())
	op1=al2.biases[feaStr].assign(feed_x)
	op2=al2.weights[feaStr].assign(feed_y)
					
	with tf.Session() as sess:
		sess.run(init)
		global step
		global data_queue
		global lr
		lr = 1e-7
		step = 0

		# Keep training until reach max iterations
		saver.restore(sess, "Data/tf_alex_pre_ccnn_model_iter285001.ckpt-285001")
		step = 1
		biasesInd    = 45
		weightsInd = 35
		myvar= tf.trainable_variables()
		#myvar = [v for v in vv if "conv" in v.name]
		len2 = len(myvar)
		myvarlist = [myvar[weightsInd], myvar[biasesInd]]
		var=tf.gradients(cost, myvarlist)
		lastvar = var

		for v in range(20):
			ts=time.time()
			centers, lab1=val_batch(n_classes, ss=0)
			rn.shuffle(seq1)
			cfeat = sess.run(pred, feed_dict={X:centers, keep_prob:1.})
			count1 = np.zeros((n_classes))
			bct = 0
			acc_batch_x = np.zeros((batch_size, 224,224,3))
			acc_batch_y = np.zeros((batch_size, n_classes))
			for k in range(0, vlen1, batch_size):
				epcho1=np.floor((step*batch_size)/vlen1)
				#============Extract feature==================
				batch_sx, batch_ys = data_queue.get()
				fea1 = sess.run(pred, feed_dict={X:batch_sx, keep_prob:1.})			
				gtlab[k:k+batch_size] = batch_ys.argmax()
				b=batch_size
				b2 = 25
				p_lab=np.zeros((b))
				p_lab2=np.zeros((b, n_classes))
				p_err=np.zeros((b))
				for j in range(b):
					diff1 = pow(pow(fea1[j,:] - cfeat, 2.0),0.5)
					diff1 = diff1.sum(axis=1)
					p_lab[j] = diff1.argmin()
					p_lab2[j, int(p_lab[j])] = 1
					count1[int(p_lab[j])] += 1
					p_err[j] = min(diff1)
					
				predicts[k:k+b] = p_lab
				# Use the most realible set to updatep
				acc_ind=sorted(range(len(p_err)), key=lambda k: p_err[k])
				myidn = acc_ind[0:b2]
				acc_batch_x[bct:bct+b2,:,:,:] =  batch_sx[myidn,:,:,:]
				acc_batch_y[bct:bct+b2,:] = p_lab2[myidn,:]
				bct = bct + b2
				
				#~ acc_batch_x = batch_sx
				#~ acc_batch_y = p_lab2
				perrave=p_err.sum(axis=0)/batch_size

				if (bct >= 50):
					bct = 0
					#===========Update network====================
					#~ sess.run(optimizer, feed_dict={X:batch_sx, Y:p_lab2, keep_prob:dropout, learning_rate:lr})
					sess.run(optimizer, feed_dict={X:acc_batch_x, Y:acc_batch_y, keep_prob:dropout, learning_rate:lr})
					
					#===========Update center======================
					var_grad = sess.run(var, feed_dict={X:batch_sx, Y: p_lab2, keep_prob:1.})
					allvar = sess.run(myvar, feed_dict={X:batch_sx, Y: p_lab2, keep_prob:1.})
					if (k%2==1):
						sess.run(op1, feed_dict={feed_x:allvar[biasesInd]  + lr*lastgrad[1]})
						sess.run(op2, feed_dict={feed_y:allvar[weightsInd]  + lr*lastgrad[0]})
						fea1 = sess.run(pred, feed_dict={X:batch_sx, keep_prob:1.})
						sess.run(op1, feed_dict={feed_x:allvar[biasesInd]})
						sess.run(op2, feed_dict={feed_y:allvar[weightsInd]})
					
					for j in range(b):
						idx1 = int(p_lab[j])
						lr2 = 1.0/count1[idx1] 
						cfeat[idx1, :] = (1-lr2)*cfeat[idx1, :] + lr2*fea1[j,:]

					lastgrad = var_grad

				#for j in range(n_classes):
					#cfeat[j, :] = cfeat[j, :] + lr*var_grad
					
				if step % display_step == 1:
					# 计算损失值
					loss = sess.run(cost, feed_dict={X: batch_sx, Y: batch_ys, keep_prob: 1.})
					acc = sess.run(accuracy, feed_dict={X: batch_sx, Y: batch_ys, keep_prob: 1.})
					acc2val = sess.run(acc2, feed_dict={X: batch_sx, Y: p_lab2, keep_prob: 1.})
					#info=sess.run(info2,feed_dict={X:batch_xs, keep_prob:1.})
					#print(info)
					print "Iter=" + str(step*batch_size) + "/epcho=" + str(np.floor((step*batch_size)/vlen1)) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy=" + "{:.5f}".format(acc) + ", lr=" + str(lr) + ", err1=" + "{:.5f}".format(perrave)
					#print "This stage has " + str(acc_batch_y.shape[0]) + " samples to be used to update the network."
				step += 1
			#input()
			nmi = mi(predicts, gtlab)
			td = time.time()-ts
			print("NMI to mini-batch kmeans is " + str(nmi) + ". Time: " + str(td) + " sec.")
			
		#print("Model saved in file: %s" % save_path)
	threading.Thread._Thread__stop(load_thread)

def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        global load_thread
        
        threading.Thread._Thread__stop(load_thread)
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
	trainer()
