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

def compute_euclidean_distances(x, y, w=None):
	d = tf.square(tf.subtract(x, y))
	if w is not None:
		d = tf.transpose(tf.multiply(tf.transpose(d), w))
	d = tf.sqrt(tf.reduce_sum(d))
	return d

class Embedding:

	def __init__(self):
		pass

	def construct(self, x):
		dim = x.get_shape()[3].value

		with tf.variable_scope('conv1'):
			out = 32
			w = weight_variable([5, 5, dim, out])
			b = bias_variable([out])
			h = max_pool_2x2(tf.nn.relu(conv2d(x, w) + b))
			dim = out
			x = h

		with tf.variable_scope('conv2'):
			out = 64
			w = weight_variable([5, 5, dim, out])
			b = bias_variable([out])
			h = max_pool_2x2(tf.nn.relu(conv2d(x, w) + b))
			dim = out
			x = h

		with tf.variable_scope('conv3'):
			out = 128
			w = weight_variable([3, 3, dim, out])
			b = bias_variable([out])
			h = max_pool_2x2(tf.nn.relu(conv2d(x, w) + b))
			dim = out
			x = h

		return h