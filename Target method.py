# This is a test of a method of constructing local targets for optimization
# due to Fong, Spivak and Tuyéras that is composable and which is described in
# their paper Backprop as Functor: A compositional perspective on supervised learning
#
# This method involves constructing elementwise error functions e(x,y) comparing
# the output of a layer, x, with the target for the layer y and the elementwise
# application of the inverse of the derivative of these functions to the 
# gradient with respect to what one wants to use as a target.
#
# With the squared Hellinger distance we have e(x,y)=(sqrt(x)-sqrt(y))^2 so that
# de(x,y)/dy = (sqrt(x)-sqrt(y))/sqrt(x) = 1 - sqrt(y/x). If 1 - sqrt(y/x)=z then
# y/x=±(1-z)^2 so that y = ±x(1-z)^2.
#
# We consider the solution x(1-z)^2, so that we obtain as target for x x(1-grad_x(loss))^2.
#
# This achieves 98.34%, 98.26, similar to gradient descent with cross entropy
# or squared Hellinger distance, but requires very many iterations.

import numpy as np
import tensorflow as tf

#Parameters
training_iters = 6000000
batch_size = 256

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

targets = tf.Variable(tf.zeros([batch_size, 1200]), dtype=tf.float32)

x1 = tf.nn.relu(tf.matmul(x, W0) + b0)
x2 = tf.nn.relu(tf.matmul(x1, W1) + b1)
x3 = tf.nn.relu(tf.matmul(x2, W2) + b2)	
softmax_output = tf.nn.softmax(x3)
	
final_error = tf.reduce_mean((tf.sqrt(softmax_output) - tf.sqrt(y))**2)

compute_targets = tf.assign(targets, x2*(tf.gradients(final_error, x2)[0]-1)**2)

second_error = tf.reduce_mean(tf.abs(tf.stop_gradient(targets) - x2))

minimizer1 = tf.train.GradientDescentOptimizer(0.05).minimize(final_error, var_list=[W2, b2])
minimizer2 = tf.train.GradientDescentOptimizer(0.01).minimize(second_error)

minimizers = [minimizer1, compute_targets, minimizer2]

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
        sess.run(compute_targets, feed_dict={x:batch_x, y:batch_y})
        sess.run(minimizers, feed_dict={x:batch_x, y:batch_y})
        if(step % 10 == 0):
            acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
            print(" Iter " + str(step*batch_size) + " , Training accuracy= " + "{:.5f}".format(acc))
        step += 1
test_accuracy = 0.0
for i in range(1,101):
    test_accuracy += sess.run(accuracy, feed_dict={x: x_test[(i-1)*100:i*100], y: y_test[(i-1)*100:i*100]})
test_accuracy = test_accuracy/100.0
print("Testing Accuracy:", test_accuracy)