# This is a comparison of three loss functions on a classification problem:
# cross entropy, quadratic error and a distance proportional to squared Hellinger
# distance, for confirming that squared Hellinger/Fisher-Rao distance
# is at least as good for comparing discrete probability distributions as cross entropy,
# which, with the choices of parameters chosen seems to be the case.

import numpy as np
import tensorflow as tf

#Parameters
training_iters = 1500000
batch_size = 128

sess = tf.Session()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, [-1, 28*28])
x_test = np.reshape(x_test, [-1, 28*28])
y_train = sess.run(tf.one_hot(y_train, 10))
y_test = sess.run(tf.one_hot(y_test, 10))

x = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])
 
W0 = tf.Variable(tf.truncated_normal([784, 1200], stddev=0.01, dtype=tf.float32))
b0 = tf.Variable(np.zeros(1200),dtype=tf.float32)

W1 = tf.Variable(tf.truncated_normal([1200, 1200], stddev=0.01, dtype=tf.float32))
b1 = tf.Variable(np.zeros(1200), dtype=tf.float32)

W2 = tf.Variable(tf.truncated_normal([1200, 10], stddev=0.01, dtype=tf.float32))
b2 = tf.Variable(np.zeros(10), dtype=tf.float32)

x1 = tf.nn.relu(tf.matmul(x, W0) + b0)
x2 = tf.nn.relu(tf.matmul(x1, W1) + b1)
x3 = tf.nn.relu(tf.matmul(x2, W2) + b2)
softmax_output = tf.nn.softmax(x3)
	
final_error = tf.reduce_mean((tf.sqrt(softmax_output) - tf.sqrt(y))**2)
#Works well with learning rate 0.1 and 1.5 million iterations
#and achieves error rates 98.34%, 98.31%, 98.37%, 79.16% 58.68%.
#Halving the learning rate to 0.05 works better
#and achieves accuracies of 98.05, 98.16, 98.23%, 98.02%, 98.20%, 98.15%

#final_error = tf.reduce_mean((softmax_output - y)**2)
#Works well with learning rate 0.01 and 1.5 million iterations
#and achieves accuracies of 97.57%, 88.39%, 97.54%

#final_error = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=x3)
#Works well with learning rate 0.00001 and 1.5 million iterations
#and achieves accuracies of 98.03%, 98.09%, 87.96%, 97.77%

#final_error = tf.reduce_mean(tf.abs(softmax_output - y))
#Works well with learning rate 0.05 and 1.5 million iterations
#and achieves accuracies of 98.03%, 98.02

minimizer = tf.train.GradientDescentOptimizer(0.05).minimize(final_error)

correct_pred = tf.equal(tf.argmax(softmax_output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess.run(tf.global_variables_initializer())	

step=1
while step*batch_size < training_iters:
    perm = np.arange(x_train.shape[0])
    np.random.shuffle(perm)
    images = x_train[perm]
    labels = y_train[perm]
    index_in_epoch = 0
    while index_in_epoch + batch_size < x_train.shape[0]:
        start = index_in_epoch
        index_in_epoch += batch_size
        end = index_in_epoch
        batch_x = images[start:end]
        batch_y = labels[start:end]
        sess.run(minimizer, feed_dict={x:batch_x, y:batch_y})
        if(step % 10 == 0):
            acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
            print(" Iter " + str(step*batch_size) + " , Training accuracy= " + "{:.5f}".format(acc))
        step += 1
test_accuracy = 0.0
for i in range(1,101):
    test_accuracy += sess.run(accuracy, feed_dict={x: x_test[(i-1)*100:i*100], y: y_test[(i-1)*100:i*100]})
test_accuracy = test_accuracy/100.0
print("Testing Accuracy:", test_accuracy)