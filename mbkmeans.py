import time
import cv2
import numpy as np
import time
import random as rn
from sklearn.metrics.cluster import normalized_mutual_info_score as mi
from sklearn import metrics
import alex_fcn_clustering as afc
from operator import itemgetter 

mean1= (104.0 ,117.0 ,123.0)

def get_filename(prefix, X):
	fn1 = X.replace('\n','')
	fn2 = fn1.split(' ')
	
	fnx = prefix+fn2[0]
	fnx = fnx.replace(" ","")
	label1 = int(fn2[1])
	imguri = fnx
	return imguri, label1

def get_batch(X, prefix, n_classes):
	global mean1
	bs = len(X)
	data = np.zeros((bs, 224,224,3))
	label1 = np.zeros((bs, n_classes))
	
	for i in range(bs):
		fn, lab = get_filename(prefix, X[i])
		im = cv2.imread(fn)
		im = np.asarray(im)
		im = cv2.resize(im, (256,256)) - mean1
		label1[i, lab] = 1
		data[i,:,:,:]=im[16:240, 16:240, :]
	return data, label1


def bkm(fn, prefix, k, b, n_classes, maxiter = 10):
		
	with open(fn) as fx:
		content = fx.readlines()
	glen1 = len(content)
	seq1 = range(glen1)
	rn.shuffle(seq1)
	ls = seq1[0:k]
	cidx = itemgetter(*ls)(content)
	centers, c_lab = get_batch(cidx, prefix, n_classes)
	print "Get centers' feature!"
	cfeat = afc.getFeat(centers, True, o_layer="fcb")
	print(cfeat.shape)
	rn.shuffle(seq1)
	
	predicts = np.zeros((glen1))
	groundtruths=np.zeros((glen1))
	iters = 0
	step = (glen1/b) / 10
	print("Step is "+str(step))
	
	preg=0
	while(maxiter>iters):
		count1 = np.zeros((n_classes))
		for i in range(0, glen1, b):
			ls = seq1[i:i+b]
			X = itemgetter(*ls)(content)
			data, lab1 = get_batch(X, prefix, n_classes)
			groundtruths[i:i+b] = lab1.argmax()
			# find nearest center and record the count 
			fea1 = afc.getFeat(data, False, o_layer="fcb")
			p_lab=np.zeros((b))
			for j in range(b):
				diff1 = pow(pow(fea1[j,:] - cfeat, 2.0),0.5)
				diff1 = diff1.sum(axis=1)
				p_lab[j] = diff1.argmin()
				count1[int(p_lab[j])] += 1
			predicts[i:i+b] = p_lab
			
			# update the center by SGD
			for j in range(b):
				idx1 = int(p_lab[j])
				lr = 1.0/count1[idx1] 
				cfeat[idx1, :] = (1-lr)*cfeat[idx1, :] + lr*(fea1[j,:])
		print(".")
		iters+=1
		nmi = mi(predicts, groundtruths)
		print("Iter " + str(iters) + " where NMI is " + str(nmi))
	
	return nmi

	
if __name__ == '__main__':
	#~ prefix='/home/jess/Disk1/ilsvrc12/ILSVRC2012_img_val/'
	#~ fn='/home/jess/Disk1/ilsvrc12/label/val.txt'
	#~ n_classes = 1000
	#~ b=50
	#~ k=1000
	prefix='Data/places2/val_large/'
	fn='Data/places2/label/places365_val.txt'
	n_classes = 365
	b=50
	k=365
	ts=time.time()
	nmi=bkm(fn, prefix, k, b, n_classes, maxiter=1)
	td = time.time()-ts
	print("NMI to mini-batch kmeans is " + str(nmi) + ". Time: " + str(td) + " sec.")
	import io
	
	ff=open("mbkmeans2d_log.txt","w")
	ff.write("NMI to mini-batch kmeans is " + str(nmi) + ". Time: " + str(td) + " sec.\n")
	ff.close()
