import Embedding
import Classifier
import tensorflow as tf

class NoEmbeddingClassifier:

	def __init__(self):
		pass

	def construct(self):
		# Input and label placeholders
		with tf.variable_scope('input'):
			self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
			self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		
		with tf.variable_scope('embedding') as scope:
			e = Embedding.Embedding()
			self.o = e.construct(self.x)
			
		with tf.variable_scope('classifier'):
			c = Classifier.Classifier()
			self.y, self.accuracy = c.construct(self.o, self.y_, self.keep_prob)
		
		with tf.variable_scope('class_loss'):
			self.class_loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

	def train(self, data_generator, batch_size=50, iterations=100, log_freq=5, keep_prob=1.0):
		train_step = tf.train.AdamOptimizer().minimize(self.class_loss)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for i in range(iterations):
				batch = data_generator.train(batch_size)

				if i % log_frequency == 0:
					loss, acc = sess.run([self.class_loss, self.accuracy], 
						feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
					print('iteration %d, training loss %g, training accuracy %g' % (i, loss, acc))
					validation_set = data_generator.validation()
					v_loss, v_acc = sess.run([self.class_loss, self.accuracy], 
						feed_dict={self.x: validation_set[0], self.y_: validation_set[1], self.keep_prob: 1.0})
					print('iteration %d, validation loss %g, validation accuracy %g' % (i, v_loss, v_acc))

				train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: keep_prob})

			test_set = data_generator.test()
			t_loss, t_acc = sess.run([self.class_loss, self.accuracy],
				feed_dict={self.x: test_set[0], self.y_: test_set[1], self.keep_prob: 1.0})
			print('test loss %g, test accuracy %g' % (t_loss, t_acc))

if __name__ == "__main__":
	model = NoEmbeddingClassifier()
	model.construct()
	model.train()