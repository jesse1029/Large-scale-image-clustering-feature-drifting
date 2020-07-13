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
import math
import pdb
mndata = MNIST('../DCN/python-mnist/data')
train_x, train_y=mndata.load_training()
train_x=np.array(train_x)
train_y=np.array(train_y)
print train_x.shape
print train_y.shape


# 定义网络超参数

batch_size = 6000
batch_size_to_be_updated = 10
b2 = 1

training_iters = batch_size*4 * 200000
display_step = 600

n_classes = 10# 标签的维度
dropout = 0.5 # Dropout 的概率
# 定义网络参数
n_input = (224,224,3) # 输入的维度
train_x=np.reshape(train_x,(60000,28,28))
avg_num = batch_size_to_be_updated / n_classes


x = tf.placeholder(tf.float32, [None, 28,28,1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

vlen1 = len(train_x)
seq1 = range(vlen1)
rn.shuffle(seq1)
batch_num = 0

isUsed = np.zeros((60000), dtype=bool)

def val_batch(bs, ss=-1):
	global batch_num
	size = 28, 28
	len1 = len(train_x)
	data = np.zeros((bs,28,28,1))
	sdata = np.zeros((bs), dtype=int)
	label1 = np.zeros((bs, n_classes));
	st = bs*batch_num if ss==-1 else bs*ss
	cnt = 0
	vbx = 0
	global seq1
	
	global phase
	for k in range(bs):
		idx1 = seq1[(k+st) % len1]
		#~ im = cv2.adaptiveThreshold(np.array(train_x[idx1,:,:], dtype = np.uint8),255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
		im = np.resize(train_x[idx1,:,:], (28,28,1)) 
		label1[cnt, train_y[idx1]] = 1
		data[cnt,:,:]=im
		sdata[cnt]=idx1
		cnt = cnt + 1
	batch_num=(batch_num+1) % len1
	return data, label1, sdata

	

#async load data
with tf.device('/cpu:0'):
	data_queue = Queue(20)

	def load_data():
		while True:
			if Queue.qsize(data_queue)<20:
				#print 'loading next batch...'+str(Queue.qsize(data_queue))
				x,y ,sx= val_batch(batch_size)
				data_queue.put((x,y,sx))
	load_thread = threading.Thread(target=load_data)
	load_thread.deamon=True


def getInd(acc_ind, p_count, p_lab, b2, b, bidx):
	global n_classes
	global avg_num
	ct = 0
	myind = np.zeros((b2), dtype=int)
	for i in range(b):
		c_lab = p_lab[acc_ind[i]]
		isFound = 0
		if (p_count[c_lab]<avg_num+1 & isUsed[bidx[i]] == False):
			isFound=1
			myind[ct] = acc_ind[i]
			isUsed[bidx[i]] =  True
			ct = ct+1
			if (ct>=b2):
				return myind
	return acc_ind[0:b2]
	

def getBatchFeature(sess, batch_sx, feaLen, fcb):
	zst= 0
	fea1 = np.zeros((batch_size, feaLen))
	for z in range(0, batch_size, 100):
		fea1[z:z+100,:] = sess.run(fcb, feed_dict={X:batch_sx[z:z+100,:,:,:], keep_prob:1.})
		zst = zst + 100
	fea1[zst:batch_size,:] = sess.run(fcb, feed_dict={X:batch_sx[zst:batch_size,:,:,:], keep_prob:1.})
	return fea1

def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        global load_thread
        threading.Thread._Thread__stop(load_thread)
        sys.exit(0)
        
signal.signal(signal.SIGINT, signal_handler)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config=tf.ConfigProto(gpu_options=gpu_options)

X = tf.placeholder("float", [None, 28, 28,1])
Y = tf.placeholder("float", [None, n_classes])	

ttk = time.time()
with tf.Session(config=config) as sess:
	
	fcb, pred = al.lenet_ccnn(X,  keep_prob)
	rec=al.lenet_ccnn_rec(X, keep_prob)
	c2=pow(tf.reduce_sum(pow(X-rec, 2.0),[0,1,2,3]),0.5)
	optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(c2)
	
	#==========Get Init Center====================@
	vlen2=vlen1+(batch_size-(vlen1%batch_size))
	saver = tf.train.Saver()		
	gtlab = np.zeros((vlen2))
	predicts = np.zeros((vlen2))

	step = 1
	
	sess.run(tf.initialize_all_variables())
	lastgrad=[]
	# Total epochs we have to reach.
	allRuntime=0
	
	from matplotlib import pyplot as plt
	
	centers, lab1, ssx=val_batch(n_classes, ss=0)
	idx2 = [0,1,2,3,4,5,7,101,55,66]
	idx2=[1, 3, 5, 7, 2, 0, 66, 101, 55, 4]
	cct=0
	lab1 = np.zeros((10, n_classes));
	for i in idx2:
		im = train_x[i,:,:]
		im = np.resize(im, (28,28,1)) 
		centers[cct,:,:,:] = im
		lab1[cct, train_y[i]] = 1
		cct=cct+1
	
	plt.subplot(331),plt.imshow(centers[0,:,:,0],'gray'),plt.title('label 0')
	plt.subplot(332),plt.imshow(centers[1,:,:,0],'gray'),plt.title('label 1')
	plt.subplot(333),plt.imshow(centers[2,:,:,0],'gray'),plt.title('label 2')
	plt.subplot(334),plt.imshow(centers[3,:,:,0],'gray'),plt.title('label 3')
	plt.subplot(335),plt.imshow(centers[4,:,:,0],'gray'),plt.title('label 4')
	plt.subplot(336),plt.imshow(centers[5,:,:,0],'gray'),plt.title('label 5')
	plt.subplot(337),plt.imshow(centers[6,:,:,0],'gray'),plt.title('label 6')
	plt.subplot(338),plt.imshow(centers[7,:,:,0],'gray'),plt.title('label 7')
	plt.subplot(339),plt.imshow(centers[8,:,:,0],'gray'),plt.title('label 8')

	#~ plt.show()
	
	
	
	#============================================
	#====== Pretraining....==========================
	#~ saver = tf.train.Saver()		
	#~ divid0 = False
	#~ divid1 = False
	#~ divid2 = False
	#~ lr = 1e-3
	#~ tmpBatch = batch_size
	#~ batch_size = 64
	#~ load_thread = threading.Thread(target=load_data)
	#~ load_thread.deamon=True
	#~ load_thread.start()	
	#~ 
	#~ for v in range(10000):
		#~ ts=time.time()
		#~ rn.shuffle(seq1)
		#~ featLen = 32
		#~ count1 = np.zeros((n_classes))
		#~ bct = 0
		#~ 
		#~ for k in range(0, vlen2, batch_size):
			#~ epcho1=np.floor((step*batch_size)/vlen1)
			#~ #============Extract feature==================
			#~ batch_sx, batch_ys,batch_ssx = data_queue.get()				
			#~ sess.run(optimizer2, feed_dict={X:batch_sx, keep_prob:dropout, learning_rate:lr})
			#~ if step % display_step == 1:
				#~ # 计算损失值
				#~ cc2 = sess.run(c2, feed_dict={X: batch_sx, keep_prob: 1.})
				#~ if (cc2 <1900) & (divid0==False):
					#~ lr = lr/10
					#~ divid0 = True
					#~ print "ohohoh.......>!!!!!!!!!!!!!! it's converged now!! learning rate will be divided by 10 first time"
				#~ if (cc2 <1200) & (divid1==False):
					#~ lr = lr/10
					#~ divid1 = True
					#~ print "ohohoh.......>!!!!!!!!!!!!!! it's converged now!! learning rate will be divided by 10"
				#~ if (cc2<500) & (divid2 ==False):
					#~ lr=lr/10
					#~ divid2 = True
					#~ print "ohohoh.......>!!!!!!!!!!!!!! it's converged now!! learning rate will be divided by 10 again"
				#~ if (cc2<10):
					#~ save_path = saver.save(sess, "Data/tf_mnist_pretrained.ckpt")
					#~ print("Lower error value is reached...! %s" % save_path)
					#~ 
					#~ 
				#~ print "Pretraining... Iter=" + str(step*batch_size) + "/epcho=" + str(v) +", lr=" + str(lr) + ", Time=" + str(time.time()-ttk)+", Rec.err="+str(cc2)
				#~ ttk = time.time()
				#~ 
			#~ 
			#~ step += 1
	#~ 
	#~ save_path = saver.save(sess, "Data/tf_mnist_pretrained.ckpt")
	#~ print("Model saved in file: %s" % save_path)
	#~ #============================================

	#~ data_queue.queue.clear()
	#~ threading.Thread._Thread__stop(load_thread)
	#~ batch_size= tmpBatch
	#====== Clustering fine-tune.==========================
	
	
	
	step=0
	batch_num = 0
	load_thread = threading.Thread(target=load_data)
	load_thread.deamon=True
	load_thread.start()	
	
	#~
	lr = 1e-5

	
	weightsInd = 4
	biasesInd = 11
	
	myvar= tf.trainable_variables()
	len2 = len(myvar)
	myvarlist = [myvar[weightsInd], myvar[biasesInd]]
	B1 = tf.placeholder("float",al.biases['bc10'].get_shape().as_list())
	updateB=  al.biases['bc10'].assign(B1)
	W1 = tf.placeholder("float", al.weights['wc10'].get_shape().as_list())
	updateW=al.weights['wc10'].assign(W1)
	
	
	trainvarlist = [myvar[weightsInd], myvar[biasesInd], myvar[6], myvar[13]]
	#~ trainvarlist = myvar
	
	rec=al.lenet_ccnn_rec(X, keep_prob)
	c2=pow(tf.reduce_sum(pow(X-rec, 2.0),[0,1,2,3]),0.5)
	cost = -tf.reduce_sum(Y*tf.log(pred))
	
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, var_list=trainvarlist)
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	cfeat = sess.run(fcb, feed_dict={X:centers, keep_prob:1.})
	
	for kk in range(len2):
		print str(kk) +"th data is " + str(myvar[kk].get_shape().as_list()) + " with " + str(type(myvar[kk]))

	varxx=tf.gradients(cost, myvarlist)
	#~ print varxx
	acc_batch_x = np.zeros((batch_size_to_be_updated, 28,28,1))
	acc_batch_y = np.zeros((batch_size_to_be_updated, n_classes))
	count1 = np.zeros((n_classes))
	
	sess.run(tf.initialize_all_variables())
	saver.restore(sess, "Data/tf_mnist_pretrained.ckpt")
	
	cfeat = sess.run(fcb, feed_dict={X:centers, keep_prob:1.})
	bct=0
	b=batch_size
	featLen = cfeat.shape[1]

	
	fea1 = np.zeros((batch_size, featLen))
	copp=0
	copp_cc=0
	pred_str1=""
	gt_str1= ""
	p_count = np.zeros((n_classes), dtype=int)
	losshist = np.zeros((1000000))
	lossct = 0
	
	for i in range(60000):
		isUsed[i]=False
	
	for v in range(10000):
		ts=time.time()
		rn.shuffle(seq1)
		cfeat = sess.run(fcb, feed_dict={X:centers, keep_prob:1.})
		count1 = np.zeros((n_classes))
		
		ttk = time.time()
		lastgrad=[]
		hasBeenUpdated = 0
		
		if (v%100)==0:
			for i in range(60000):
				isUsed[i]=False
		
		for k in range(0, vlen2, batch_size):
			epcho1=v
			#============Extract feature==================
			batch_sx, batch_ys, bidx = data_queue.get()
			
			if (lastgrad != []):
				sess.run(updateB,  feed_dict={B1:lastvar[biasesInd]})
				sess.run(updateW, feed_dict={W1:lastvar[weightsInd]})
				fea1 = getBatchFeature(sess, batch_sx, featLen, fcb)
				sess.run(updateB, feed_dict={B1:allvar[biasesInd]})
				sess.run(updateW, feed_dict={W1:allvar[weightsInd]})
				lastgrad = var_grad + lastgrad
			else:
				fea1 = getBatchFeature(sess, batch_sx, featLen, fcb)
				
			
			gtlab[k:k+batch_size] = batch_ys.argmax(axis=1)
			b=batch_size
			p_lab=np.zeros((b),dtype=int)
			p_lab2=np.zeros((b, n_classes),dtype=int)
			p_err=np.zeros((b))
			
			
			for j in range(b):
				diff1 = pow(fea1[j,:] - cfeat, 2.0)
				diff1 = pow(diff1.sum(axis=1),0.5)
				p_lab[j] = diff1.argmin()
				p_lab2[j, int(p_lab[j])] = 1
				count1[int(p_lab[j])] += 1
				p_err[j] = min(diff1)
				
			predicts[k:k+b] = p_lab
			# Use the most realible set to update network parameters
			acc_ind=sorted(range(len(p_err)), key=lambda z: p_err[z])
			myidn =  getInd(acc_ind, p_count, p_lab, b2, b, bidx)
			acc_batch_x[bct:bct+b2,:,:,:] = batch_sx[myidn,:,:,:]
			#~ acc_batch_y[bct:bct+b2,:] = p_lab2[myidn,:]
			acc_batch_y[bct:bct+b2,:] = batch_ys[myidn,:]
			
			perrave=p_err.sum(axis=0)/batch_size
			cop = batch_ys.argmax(axis=1)
			bct = bct + b2
			
			for z in range(b2):
				pred_str1 = pred_str1 + str(p_lab[myidn[z]])+","
				gt_str1 = gt_str1 + str(cop[myidn[z]])+","
				p_count[p_lab[myidn[z]]]=p_count[p_lab[myidn[z]]]+1
			copp = copp+ (cop[myidn]==p_lab[myidn]).sum(axis=0)
			copp_cc=copp_cc+1
			
			#When the size of the collected training sample is larger than batch size, performing network updating -*-
			if (bct >= batch_size_to_be_updated):
				bct = 0
				p_count = np.zeros((n_classes), dtype=int)
				
				#===========Update network====================
				if (hasBeenUpdated==0):
					lastvar = sess.run(myvar, feed_dict={keep_prob:1.})
					hasBeenUpdated = 1
				sess.run(optimizer, feed_dict={X:acc_batch_x, Y:acc_batch_y, keep_prob:dropout, learning_rate:lr})
				var_grad = sess.run(varxx, feed_dict={ X:acc_batch_x, Y:acc_batch_y, keep_prob:1.})
				allvar = sess.run(myvar, feed_dict={keep_prob:1.})
				acc_batch_y = np.zeros((batch_size_to_be_updated, n_classes))
				
				lastgrad = var_grad
				#===========Update center======================

			for j in myidn:
				idx1 = int(p_lab[j ])
				lr2 = 1.0/count1[idx1] 
				cfeat[idx1, :] = (1-lr2)*cfeat[idx1, :] + lr2*fea1[j, :]
				
			if step % display_step == 1:
				# 计算损失值
				loss = sess.run(cost, feed_dict={X: acc_batch_x[0:bct,:,:,:], Y: acc_batch_y[0:bct], keep_prob: 1.})
				acc = sess.run(accuracy, feed_dict={X: acc_batch_x[0:bct,:,:,:], Y: acc_batch_y[0:bct], keep_prob: 1.})
				print "Iter=" + str(step*batch_size) + "/epcho=" + str(np.floor((step*batch_size)/vlen1)) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy=" + "{:.5f}".format(acc) + ", lr=" + str(lr) + ",time"+str(time.time()-ttk)
				ttk = time.time()
				print "there are " + str(copp) +"/"+str(copp_cc) + " samples are estimated as corrected!!"

				losshist[lossct] = loss
				lossct = lossct+1
				copp=0
				copp_cc=0
				pred_str1=""
				gt_str1=""
			
				
			step += 1
		if (v>300):
			if (v%300)==0 | (v%3000)==0:
				lr=lr/10
			
		if (v % 100)==0:
			nmi = mi(predicts[0:k+b], gtlab[0:k+b])
			td = time.time()-ts
			allRuntime=allRuntime+td
			print("\033[1;31m"+str(v)+"th epoch's NMI to mini-batch kmeans is " + str(nmi) + ". Time: " + str(td) + " sec.\033[0m\n")
			
	save_path = saver.save(sess, "Data/tf_mnist_" +  str(step) + ".ckpt")
threading.Thread._Thread__stop(load_thread)

