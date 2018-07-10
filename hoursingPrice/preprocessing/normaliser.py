
import tensorflow as tf
import numpy as np


def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_out = []
    for item in arr:
        out = np.divide(np.subtract(item, arr_min), np.subtract(arr_max, arr_min))
        arr_out = np.append(arr_out, np.array(out))
    return arr_out


def model(x, b, a):
    # linear regression is just b*x + a, so this model line is pretty simple
    return tf.multiply(x, b) + a


loss = tf.multiply(tf.square(Y - y_model), 0.5)

# construct an optimizer to minimize cost and fit line to mydata
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

for i in range(500):
    for (x, y) in zip(trX, trY):
        output = sess.run(train_op, feed_dict={X: x, Y: y})