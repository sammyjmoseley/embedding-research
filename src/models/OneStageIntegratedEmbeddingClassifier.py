# 1-stage integrated embedding
# Use embedding as regularizer

from models import Embedding
from models import Classifier
import tensorflow as tf

def compute_euclidean_distances(x, y, w=None):
	d = tf.square(tf.subtract(x, y))
	if w is not None:
		d = tf.transpose(tf.multiply(tf.transpose(d), w))
	d = tf.sqrt(tf.reduce_sum(d))
	return d

class OneStageIntegratedEmbeddingClassifier:

	def __init__(self):
		pass

	def construct(self, alpha=1):
		# Input and label placeholders
		with tf.variable_scope('input'):
			self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
			self.xp = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='xp')
			self.xn = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='xn')
			self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		
		with tf.variable_scope('embedding') as scope:
			e = Embedding.Embedding()
			self.o = e.construct(self.x)
			scope.reuse_variables()
			self.op = e.construct(self.xp)
			self.on = e.construct(self.xn)
			
		with tf.variable_scope('distances'):
			self.dp = compute_euclidean_distances(self.o, self.op)
			self.dn = compute_euclidean_distances(self.o, self.on)
			self.logits = tf.nn.softmax([self.dp, self.dn], name="logits")
		
		with tf.variable_scope('embed_loss'):
			self.embed_loss = tf.reduce_mean(tf.pow(self.logits[0], 2))

		with tf.variable_scope('classifier'):
			c = Classifier.Classifier()
			self.y, self.accuracy = c.construct(self.o, self.y_, self.keep_prob)
		
		with tf.variable_scope('class_loss'):
			self.class_loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

		with tf.variable_scope('weighted_loss'):
			self.weighted_loss = self.class_loss + alpha * self.embed_loss

	def train(self, data_generator, batch_size=50, iterations=100, log_freq=5, keep_prob=1.0):
		train_step = tf.train.AdamOptimizer().minimize(self.weighted_loss)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			# Stage 1: Embedding
			for i in range(iterations):
				triplet_batch = data_generator.triplet_train(batch_size)

				if i % log_freq == 0:
					e_loss, c_loss, w_loss, acc = sess.run([self.embed_loss, self.class_loss, self.weighted_loss, self.accuracy], 
						feed_dict={self.x: triplet_batch.get_reference(), self.xp: triplet_batch.get_positive(), self.xn: triplet_batch.get_negative(), self.y_: triplet_batch.get_reference_class(), self.keep_prob: 1.0})
					v_batch_x, v_batch_y_ = data_generator.validation()
					v_loss, v_acc = sess.run([self.class_loss, self.accuracy], 
						feed_dict={self.x: v_batch_x, self.y_: v_batch_y_, self.keep_prob: 1.0})
					print('iteration %d, embed loss %g, training loss %g, weighted loss %g, training accuracy %g, validation loss %g, validation accuracy %g' % (i, e_loss, c_loss, w_loss, acc, v_loss, v_acc))

				train_step.run(feed_dict={self.x: triplet_batch.get_reference(), self.xp: triplet_batch.get_positive(), self.xn: triplet_batch.get_negative(), self.y_: triplet_batch.get_reference_class(), self.keep_prob: keep_prob})

			t_batch_x, t_batch_y_ = data_generator.test()
			t_loss, t_acc = sess.run([self.class_loss, self.accuracy],
				feed_dict={self.x: t_batch_x, self.y_: t_batch_y_, self.keep_prob: 1.0})
			print('test loss %g, test accuracy %g' % (t_loss, t_acc))