import numpy as np
import cv2
import os
import tensorflow as tf

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# images and labels array as input
def convert_to(images, labels, name):
  num_examples = labels.shape[0]
  if images.shape[0] != num_examples:
    raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = name
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    
fn='/Data/jess/ILSVRC2012_img_val/'
fn2='/home/jess/caffe-old/data/ilsvrc12/val.txt'

with open(fn2) as fx:
	content = fx.readlines()
	
size = 256, 256
skip_step = 10
len1 = len(content)
data = np.zeros((len1/skip_step,256,256,3))
label1 = np.zeros((len1/skip_step));

cnt = 0
for k in range(0,len1-1,skip_step):
	fn1 = content[k].replace('\n','')
	fn2 = fn1.split(' ')
	#print(fn+fn2[0])
	fnx = fn+fn2[0]
	fnx = fnx.replace(" ","")
	im = cv2.imread(fnx)
	#print(type(im))
	im = np.asarray(im)
	
	im = cv2.resize(im, (256,256))
	row,col,channels = im.shape
	if k%500==0:
		print("Processing " + str(k) + " images / " + str(len1) + " images.")
		print(str(k )+ ' image where size is ' + str(col) + "x" +str(row))
		
	try:
		label1[cnt] = int(fn2[1])
		data[cnt,:,:,:]=im
		cnt = cnt + 1
	except Exception:
		pass
		
convert_to(data, label1, '/Disk3/jess/ilsvrc12_val_small.tfrecords')
	
