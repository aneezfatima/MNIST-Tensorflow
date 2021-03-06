import pandas as pd
import numpy as np
import tensorflow as tf

#Read train and test data
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#Extract labels as one-hot arrays
labels=pd.get_dummies(train.label)
#Train dataset need to have only images
del train['label']

sess = tf.InteractiveSession()
#784 pixels and 10 classes
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#initialize variables
sess.run(tf.global_variables_initializer())
y = tf.matmul(x,W) + b

#training
for _ in range(1000):
  #Batch contains 100 random images from train dataset
	rows = np.random.choice(train.index.values, 100)
	batch_xs=train.ix[rows]
	batch_ys=labels.ix[rows]
	train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#First Convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#Second convolutional layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

#Training
for i in range(20000):
	rows_50 = np.random.choice(train.index.values, 50)
	batch_xs=train.ix[rows_50]
	batch_ys=labels.ix[rows_50]
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

#Prediction
prediction=tf.argmax(y_conv,1)
ans=prediction.eval(feed_dict={x:test,keep_prob: 1.0})
a=np.asarray(ans)
np.savetxt("ans.csv", a, delimiter=",")

