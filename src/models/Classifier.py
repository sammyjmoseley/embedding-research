import tensorflow as tf

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.01)
	return tf.get_variable("weights", dtype=tf.float32, initializer=initial)

def bias_variable(shape):
	initial = tf.constant(0.01, shape=shape, dtype=tf.float32)
	return tf.get_variable("biases", dtype=tf.float32, initializer=initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')

class Classifier:

	def __init__(self):
		pass

	def construct(self, x, y_, keep_prob):
		x = tf.reshape(x, [-1, x.get_shape()[1].value * x.get_shape()[2].value * x.get_shape()[3].value])
		dim = x.get_shape()[1].value

		with tf.variable_scope('fc3') as scope:
			out = 10
			x = tf.nn.dropout(x, keep_prob=keep_prob)
			w = weight_variable([dim, out])
			b = bias_variable([out])
			self.y = tf.nn.softmax(tf.matmul(x, w) + b)
		
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		return self.y, self.accuracy