import time
import cv2
import numpy as np
import time
import random as rn
from sklearn.metrics.cluster import normalized_mutual_info_score as mi
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans, KMeans
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
		label1[i, lab-1] = 1
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
	cfeat = afc.getFeat(centers, True, o_layer="out")
	print(cfeat.shape)
	rn.shuffle(seq1)
	
	predicts = np.zeros((glen1))
	groundtruths=np.zeros((glen1))
	iters = 0
	step = (glen1/b) / 10
	print("Step is "+str(step))
	
	preg=0

	count1 = np.zeros((n_classes))
	allfea = np.zeros((glen1, 1000))
	print "Extracting features...."
	for i in range(0, glen1, b):
		ls = seq1[i:i+b]
		X = itemgetter(*ls)(content)
		data, lab1 = get_batch(X, prefix, n_classes)
		groundtruths[i:i+b] = lab1.argmax()
		# find nearest center and record the count 
		fea1 = afc.getFeat(data, False, o_layer="out")
		allfea[i:i+b]=fea1
		if (i % round(glen1/100) < b):
			print "Progress in extracting feature: " + str(i/float(glen1)*100) + "%"
	np.save("kmeans_places_val.fea", allfea)
	#~ allfea=np.load("kmeans_places_val.fea.npy")
	print "Start to cluster images..."
	k_means = KMeans(init=cfeat, n_clusters=n_classes, n_init=1)
	k_means.fit(allfea)
	predicts = k_means.labels_
	nmi = mi(predicts, groundtruths)
	
	return nmi


if __name__ == '__main__':
	prefix='Data/places2/val_large/'
	fn='Data/places2/label/places365_val.txt'
	n_classes = 365
	b=50
	k=365
	ts=time.time()
	nmi=bkm(fn, prefix, k, b, n_classes, maxiter=1)
	td = time.time()-ts
	import io
	
	print("NMI to kmeans is " + str(nmi) + ". Time: " + str(td) + " sec.")
	ff=open("kmeans2d_log.txt","w")
	ff.write("NMI to kmeans is " + str(nmi) + ". Time: " + str(td) + " sec.\n")
	ff.close()
