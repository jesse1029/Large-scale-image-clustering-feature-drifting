import tensorflow as tf
import data

# adapted from TensorFlow-Examples/alexnet.py

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

# Parameters
learning_rate = 0.001
batch_size = 256
training_iters = batch_size * 350000
display_step = 20
save_step = 100

# Network Parameters
n_input = 64*64*3               # input of each tile
n_classes = 100                 # number of jigsaw permutations
dropout = 0.8                   # Dropout, probability to keep units

keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

#async load data
from Queue import Queue
data_queue = Queue(8)

def load_data():
    while True:
        print 'loading next batch...'
        x,y = data.next_batch(batch_size)
        data_queue.put((x,y))

import threading
for i in range(4):
    load_thread = threading.Thread(target=load_data)
    load_thread.start()

def conv2d(name, l_input, w, b, k=1): # stride of 1?
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, k, k, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net_module(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 64, 64, 3])

    last = _X

    for idx in range(1, 6):
        print 'anm', idx, last
        
        # Convolution Layer
        stride = 2 if idx==1 else 1
        conv = conv2d('conv%d' % (idx), last, _weights['wc%d' % (idx)], _biases['bc%d' % (idx)], k=stride)
        if idx in (1, 2, 5):
            # Max Pooling (down-sampling)
            pool = max_pool('pool%d' % (idx), conv, k=2)
            # Apply Normalization
            norm_ = norm('norm%d' % (idx), pool, lsize=4)
            # Apply Dropout
            norm_ = tf.nn.dropout(norm_, _dropout)

            last = norm_
        else:
            last = conv

        print_activations(last)

    # Fully connected layer
    dense1 = tf.reshape(last, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation

    return dense1

# Shared layer weights & bias (?)
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),        
    'wd1': tf.Variable(tf.random_normal([4*4*256, 512])),
    'wd2': tf.Variable(tf.random_normal([9*512, 4096])),
    'wd3': tf.Variable(tf.random_normal([4096, 100])),
}

biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),    
    'bd1': tf.Variable(tf.random_normal([512])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'bd3': tf.Variable(tf.random_normal([100]))                           
}

inputs = [tf.placeholder(tf.float32, [None, n_input]) for x in range(9)]
alexes = [alex_net_module(inp, weights, biases, keep_prob) for inp in inputs]

fc_concat = tf.concat(1, alexes)
dense2 = tf.nn.relu(tf.matmul(fc_concat, weights['wd2']) + biases['bd2'], name='fc2') # Relu activation

dense3 = tf.nn.relu(tf.matmul(dense2, weights['wd3']) + biases['bd3'], name='fc3') # Relu activation
#pred = tf.matmul(dense2, weights['wd3']) + biases['bd3']
pred = dense3

y = tf.placeholder(tf.float32, [None, n_classes])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    #sess.run(init)
    #step = 1
    cpoint = tf.train.latest_checkpoint('.')
    saver.restore(sess, cpoint)
    step = int(cpoint.split('-')[1])
    print 'restored from %d' % (step)
    
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        print 'get batch...', step, (step*batch_size)
        #batch_xs, batch_ys = data_queue.get()
        batch_xs, batch_ys = data.next_batch(batch_size)

        # Fit training using batch data
        feed_dict={y: batch_ys, keep_prob: dropout}
        for inp_idx, place_x in enumerate(inputs):
            feed_dict[place_x] = [X[inp_idx] for X in batch_xs]
        sess.run(optimizer, feed_dict=feed_dict)
        
        if step % display_step == 0:
            # Calculate batch accuracy
            feed_dict={y: batch_ys, keep_prob: 1.}
            for inp_idx, place_x in enumerate(inputs):
                feed_dict[place_x] = [X[inp_idx] for X in batch_xs]
            acc = sess.run(accuracy, feed_dict=feed_dict)
            # Calculate batch loss
            feed_dict={y: batch_ys, keep_prob: 1.}
            for inp_idx, place_x in enumerate(inputs):
                feed_dict[place_x] = [X[inp_idx] for X in batch_xs]
            loss = sess.run(cost, feed_dict=feed_dict)
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
        if step % save_step == 0:
            saver.save(sess, 'jigsaw', global_step=step)
        step += 1
    print "Optimization Finished!"
        
