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
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#training
for _ in range(1000):
  #Batch contains 100 random images from train dataset
	rows = np.random.choice(train.index.values, 100)
	batch_xs=train.ix[rows]
	batch_ys=labels.ix[rows]
	train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
        
#prediction for test data
prediction=tf.argmax(y,1)
ans=prediction.eval(feed_dict={x:test})
a=np.asarray(ans)
np.savetxt("ans.csv", a, delimiter=",")
