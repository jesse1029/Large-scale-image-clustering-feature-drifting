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
display_step = 40
n_classes = 365 # 标签的维度
dropout = 0.5 # Dropout 的概率
# 定义网络参数
n_input = (224,224,3) # 输入的维度


x = tf.placeholder(tf.float32, [None, 224,224,3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
lr = 0.1
learning_rate = tf.placeholder(tf.float32)

fnv='/Disk3/jess/data_large/'
fnv2='/Data/jess/places2/label/places365_train_standard.txt'

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
		label1[cnt, int(fn2[1])-1] = 1
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
	X = tf.placeholder("float", [None, 224, 224, 3])
	sX = tf.placeholder("float", [None, 7,7])
	Y = tf.placeholder("float", [None, n_classes])	
	X2 = tf.placeholder(tf.float32,  [None, 224, 224, 3])
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.66)
	config=tf.ConfigProto(gpu_options=gpu_options)

	# 构建模型
	oldpred = al.getalex(X, keep_prob)
	sess2 = tf.Session(config=config)
	sess2.run(tf.initialize_all_variables())
	saver = tf.train.Saver()
	saver.restore(sess2, "/Data/jess/tf_pretrained.ckpt")
	xx =tf.trainable_variables()
	sess2.close()
	
	import alexccnnmodel as al2
	biasesInd    = 44
	weightsInd = 34
	B1 = tf.placeholder("float",al2.biases['bc9'].get_shape().as_list())
	updateB=  al2.biases['bc9'].assign(B1)
	
	W1 = tf.placeholder("float", al2.weights['wc9'].get_shape().as_list())
	updateW=al2.weights['wc9'].assign(W1)
	
	sess2 = tf.Session(config=config)
	sess2.run(tf.initialize_all_variables())
	fc8= al2.getccnn(X,keep_prob, xx, xx)
	saver.restore(sess2, "/Data/jess/tf_alex_pre_ccnn_model_iter285001.ckpt-285001")
	xx2 =tf.trainable_variables()
	sess2.close()
	ttk = time.time()
	with tf.Session(config=config) as sess:
		
		w10=tf.Variable(tf.random_normal([1000, n_classes], stddev=0.01))
		b10= tf.Variable(tf.random_normal([n_classes], stddev=0.01))
		pred = al2.alex_ccnn10(X, xx2, xx2, keep_prob,w10,b10, o_layer="out")
		fcb = al2.alex_ccnn10(X, xx2, xx2, keep_prob, w10, b10, o_layer="fcb")
		
		cost = -tf.reduce_sum(Y*tf.log(pred))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		
		#==========Get Init Center====================@
		#~ centers, lab1=val_batch(n_classes, ss=0)
		#~ rn.shuffle(seq1)
		saver = tf.train.Saver()		
		gtlab = np.zeros((vlen1))
		predicts = np.zeros((vlen1+(batch_size-(vlen1%batch_size))))

		lr = 1e-5
		step = 1
		len2 = len(xx2)
		myvarlist = [xx2[weightsInd], xx2[biasesInd]]
		myvar= tf.trainable_variables()
		for kk in range(len2):
			print str(kk) +"th data is " + str(xx2[kk].get_shape().as_list()) + " with " + str(type(xx2[kk]))

		varxx=tf.gradients(cost, myvarlist)
	
		print varxx
		sess.run(tf.initialize_all_variables())
		lastgrad=[]
		# Total epochs we have to reach.
		allRuntime=0
		for v in range(10):
			ts=time.time()
			centers, lab1=val_batch(n_classes, ss=0)
			rn.shuffle(seq1)
			load_thread = threading.Thread(target=load_data)
			load_thread.deamon=True
			load_thread.start()	
			
			cfeat = sess.run(fcb, feed_dict={X:centers, keep_prob:1.})
			featLen = cfeat.shape[1]
			count1 = np.zeros((n_classes))

			acc_batch_x = np.zeros((batch_size, 224,224,3))
			acc_batch_y = np.zeros((batch_size, n_classes))
			bct = 0
			vlen2=vlen1+(batch_size-(vlen1%batch_size))
			for k in range(0, vlen2, batch_size):
				epcho1=np.floor((step*batch_size)/vlen1)
				#============Extract feature==================
				batch_sx, batch_ys = data_queue.get()
				fea1 = sess.run(fcb, feed_dict={X:batch_sx, keep_prob:1.})			
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
				# Use the most realible set to update network parameters
				acc_ind=sorted(range(len(p_err)), key=lambda k: p_err[k])
				myidn = acc_ind[0:b2]
				acc_batch_x[bct:bct+b2,:,:,:] =  batch_sx[myidn,:,:,:]
				acc_batch_y[bct:bct+b2,:] = p_lab2[myidn,:]
				
				bct = bct + b2
				
				#When the size of the collected training sample is larger than batch size, performing network updating -*-
				if (bct >= batch_size):
					bct = 0
					#===========Update network====================
					sess.run(optimizer, feed_dict={X:acc_batch_x, Y:acc_batch_y, keep_prob:dropout, learning_rate:lr})
					
					#===========Update center======================
					#~ hh=input("Input any word to continue...")
					var_grad = sess.run(varxx, feed_dict={X:acc_batch_x, Y: acc_batch_y, keep_prob:1.})
					allvar = sess.run(myvar, feed_dict={X:acc_batch_x, Y: acc_batch_y, keep_prob:1.})
					if (lastgrad != []) :
						bct = 0
						sess.run(updateB,  feed_dict={B1:allvar[biasesInd]  + lr*lastgrad[1]})
						sess.run(updateW, feed_dict={W1:allvar[weightsInd]  + lr*lastgrad[0]})
						fea1 = sess.run(fcb, feed_dict={X:acc_batch_x, keep_prob:1.})
						sess.run(updateB, feed_dict={B1:allvar[biasesInd]})
						sess.run(updateW, feed_dict={W1:allvar[weightsInd]})
						#print  str(k) + "th ==> " + str(lastgrad[1][1:5])
					lastgrad = var_grad
				
				for j in myidn:
					idx1 = int(p_lab[j ])
					lr2 = 1.0/count1[idx1] 
					cfeat[idx1, :] = (1-lr2)*cfeat[idx1, :] + lr2*fea1[j, :]

				#for j in range(n_classes):
					#cfeat[j, :] = cfeat[j, :] + lr*var_grad
					
				if step % display_step == 1:
					loss = sess.run(cost, feed_dict={X: batch_sx, Y: batch_ys, keep_prob: 1.})
					acc = sess.run(accuracy, feed_dict={X: batch_sx, Y: batch_ys, keep_prob: 1.})

					print "Iter=" + str(step*batch_size) + "/epcho=" + str(v) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy=" + "{:.5f}".format(acc) + ", lr=" + str(lr) + ", err1=" + str(time.time()-ttk)
					ttk = time.time()
				
				if step % 1000==2:
					nmi2 = mi(predicts[0:k+b], gtlab[0:k+b])
					print(" Cuttent NMI to mini-batch kmeans is " + str(nmi2))
				step += 1
			#input()
			nmi = mi(predicts, gtlab)
			td = time.time()-ts
			allRuntime=allRuntime+td
			print(str(v)+"th epoch's NMI to mini-batch kmeans is " + str(nmi) + ". Time: " + str(td) + " sec.")
			save_path = saver.save(sess, "/Data/tf_place_val_step_" +  str(step) + ".ckpt")
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
