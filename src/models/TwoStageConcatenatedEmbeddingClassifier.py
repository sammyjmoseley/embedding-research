# 2-stage concatenated embedding
# Embed in parallel; (freeze if freeze_embed=True); train classifier

from models import Embedding
from models import Classifier
import tensorflow as tf
import numpy as np

def compute_euclidean_distances(x, y, w=None):
    d = tf.square(tf.subtract(x, y))
    if w is not None:
        d = tf.transpose(tf.multiply(tf.transpose(d), w))
    d = tf.sqrt(tf.reduce_sum(d))
    return d

class TwoStageConcatenatedEmbeddingClassifier:

    def __init__(self, freeze_embed=True):
        self.freeze_embed = freeze_embed

    def construct(self, softmax=True, margin=0.2):
        # Input and label placeholders
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
            self.xp = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='xp')
            self.xn = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='xn')
            self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        e = Embedding.Embedding()
        with tf.variable_scope('embedding') as scope:
            self.o = e.construct(self.x, init_dim=2)
            scope.reuse_variables()
            self.op = e.construct(self.xp, init_dim=2)
            self.on = e.construct(self.xn, init_dim=2)

        with tf.variable_scope('distances'):
            self.dp = compute_euclidean_distances(self.o, self.op)
            self.dn = compute_euclidean_distances(self.o, self.on)
            self.logits = tf.nn.softmax([self.dp, self.dn], name="logits")

        with tf.variable_scope('embed_loss'):
            if softmax:
                self.embed_loss = tf.reduce_mean(tf.pow(self.logits[0], 2))
            else:
                self.embed_loss = tf.reduce_mean(tf.maximum(tf.square(self.dp) - tf.square(self.dn) + margin, 0))

        with tf.variable_scope('classifier'):
            self.f = e.construct(self.x, init_dim=2)
            c = Classifier.Classifier()
            self.y, self.accuracy = c.construct(tf.concat([self.o, self.f], 3), self.y_, self.keep_prob)

        with tf.variable_scope('class_loss'):
            self.class_loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

    def train(self, data_generator, batch_size=50, iterations=100, log_freq=5, keep_prob=1.0, embed_iterations=100, embed_batch_size=16,
              embed_visualize=False,
              embed_visualize_size=100, only_originals=False):
        embed_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "embedding")
        embed_train_step = tf.train.AdamOptimizer().minimize(self.embed_loss, var_list=embed_train_vars)
        class_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classifier")
        if (self.freeze_embed == True):
            class_train_step = tf.train.AdamOptimizer().minimize(self.class_loss, var_list = class_train_vars)
        else:
            class_train_step = tf.train.AdamOptimizer().minimize(self.class_loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            run_name = './train/run_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            print(run_name)

            # Visualize:
            vis_batch_x, vis_batch_y_ = data_generator.get_embedding_visualization_data()
            if embed_visualize:
                vis_batch_embed = sess.run(self.o, feed_dict={self.x: vis_batch_x, self.keep_prob: 1.0})
                EmbeddingVisualizer.visualize(vis_batch_x, vis_batch_embed, vis_batch_y_, run_name+"/init")

            # Stage 1: Embedding
            for i in range(embed_iterations):
                triplet_batch = data_generator.triplet_train(embed_batch_size)

                if i % log_freq == 0:
                    loss = sess.run(self.embed_loss,
                        feed_dict={self.x: triplet_batch.get_reference(), self.xp: triplet_batch.get_positive(), self.xn: triplet_batch.get_negative()})
                    print('iteration %d, embedding loss %g' % (i, loss))

                embed_train_step.run(feed_dict={self.x: triplet_batch.get_reference(), self.xp: triplet_batch.get_positive(), self.xn: triplet_batch.get_negative()})

            if embed_visualize:
                vis_batch_embed = sess.run(self.o, feed_dict={self.x: vis_batch_x, self.keep_prob: 1.0})
                EmbeddingVisualizer.visualize(vis_batch_x, vis_batch_embed, vis_batch_y_, run_name+"/embed")

            # Stage 2: Classification
            for i in range(iterations):
                batch_x, batch_y_ = data_generator.train(batch_size, only_originals=only_originals)

                if i % log_freq == 0:
                    loss, acc = sess.run([self.class_loss, self.accuracy],
                        feed_dict={self.x: batch_x, self.y_: batch_y_, self.keep_prob: 1.0})
                    v_batch_x, v_batch_y_ = data_generator.validation()
                    v_loss, v_acc = sess.run([self.class_loss, self.accuracy],
                        feed_dict={self.x: v_batch_x, self.y_: v_batch_y_, self.keep_prob: 1.0})
                    print('iteration %d, training loss %g, training accuracy %g, validation loss %g, validation accuracy %g' % (i, loss, acc, v_loss, v_acc))

                class_train_step.run(feed_dict={self.x: batch_x, self.y_: batch_y_, self.keep_prob: keep_prob})

            if embed_visualize:
                vis_batch_embed = sess.run(self.o, feed_dict={self.x: vis_batch_x, self.keep_prob: 1.0})
                EmbeddingVisualizer.visualize(vis_batch_x, vis_batch_embed, vis_batch_y_, run_name+"/final")

            t_batch_x, t_batch_y_ = data_generator.test()
            t_loss, t_acc = sess.run([self.class_loss, self.accuracy],
                feed_dict={self.x: t_batch_x, self.y_: t_batch_y_, self.keep_prob: 1.0})
            print('test loss %g, test accuracy %g' % (t_loss, t_acc))