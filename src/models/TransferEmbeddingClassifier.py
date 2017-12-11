# Full network trained end-to-end on classification
# (No embedding component at all)

from models import Embedding
from models import Classifier
import tensorflow as tf
import numpy as np
from data_generators.SequentialDataGenerator import RotatedMNISTDataGenerator
from datetime import datetime

graph = tf.get_default_graph()

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
    d = tf.sqrt(tf.reduce_sum(d, axis=1))
    return d

class NoEmbeddingClassifier:
    def __init__(self):
        # Input and label placeholders
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
            self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        dim = 1
        init_dim = 2
        x = self.x
        with tf.variable_scope('conv1'):
            print(x.shape)
            out = init_dim
            w = weight_variable([3, 3, dim, out])
            b = bias_variable([out])
            h = max_pool_2x2(tf.nn.relu(conv2d(x, w) + b))
            dim = out
            x = h
            print(x.shape)

        with tf.variable_scope('conv2'):
            out = dim * 2
            w = weight_variable([3, 3, dim, out])
            b = bias_variable([out])
            h = max_pool_2x2(tf.nn.relu(conv2d(x, w) + b))
            dim = out
            x = h
            print(x.shape)

        with tf.variable_scope('conv3'):
            out = dim * 2
            w = weight_variable([3, 3, dim, out])
            b = bias_variable([out])
            h = max_pool_2x2(tf.nn.relu(conv2d(x, w) + b))
            dim = out
            x = h
            dim = 128
            x = tf.reshape(x, [-1, dim])
            self.convolution_embedding = tf.placeholder_with_default(x,
                                                                     x.shape,
                                                                     name="convolution_embedding")
            print(self.convolution_embedding.shape)

        with tf.variable_scope('fc1'):
            out = 10
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)

            w = weight_variable([dim, out])
            b = bias_variable([out])
            h = tf.nn.relu(tf.matmul(x, w) + b)
            dim = out
            x = h


        with tf.variable_scope('fc2'):
            out = 10
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)
            w = weight_variable([dim, out])
            b = bias_variable([out])
            self.before_softmax = tf.matmul(x, w) + b
            print(self.before_softmax.shape)
            self.y = tf.nn.softmax(self.before_softmax, dim=1)

        with tf.variable_scope('class_loss'):
            print(self.y_.shape)
            print(self.before_softmax.shape)
            self.class_loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, data_generator, batch_size=50, iterations=100, log_freq=5, keep_prob=1.0):
        train_step = tf.train.AdamOptimizer(1e-3).minimize(self.class_loss)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            run_name = './train/run_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            print(run_name)

            for i in range(iterations):
                batch_x, batch_y_ = data_generator.train(batch_size)

                if i % log_freq == 0:
                    loss, = sess.run([self.class_loss],
                                         feed_dict={self.x: batch_x, self.y_: batch_y_, self.keep_prob: 1.0})
                    v_batch_x, v_batch_y_ = data_generator.validation()
                    v_loss, v_acc = sess.run([self.class_loss, self.accuracy],
                                             feed_dict={self.x: v_batch_x, self.y_: v_batch_y_, self.keep_prob: 1.0})
                    print(
                        'iteration {}, training loss {}, validation loss {}, acc {}'.format(i, loss, v_loss, v_acc))

                train_step.run(feed_dict={self.x: batch_x, self.y_: batch_y_, self.keep_prob: keep_prob})

            t_batch_x, t_batch_y_ = data_generator.test(augment=False)
            t_loss = sess.run([self.class_loss],
                                     feed_dict={self.x: t_batch_x, self.y_: t_batch_y_, self.keep_prob: 1.0})
            print('test loss {}, test accuracy'.format(t_loss))

    def convolution_embed(self, sess, x):
        embedding, = sess.run([self.convolution_embedding],
                         feed_dict={self.x: x, self.keep_prob: 1.0})
        return embedding


class RotatedEmbeddingClassifier:
    def __init__(self, no_embedding_classifier):
        self.no_embedding_classifier = no_embedding_classifier

        # Input and label placeholders
        with tf.variable_scope('input'):
            self.x = no_embedding_classifier.x
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        dim = 1
        init_dim = 2
        x = self.x
        with tf.variable_scope('rotated_embedding'):
            with tf.variable_scope('conv1'):
                print(x.shape)
                out = init_dim
                w = weight_variable([3, 3, dim, out])
                b = bias_variable([out])
                h = max_pool_2x2(tf.nn.relu(conv2d(x, w) + b))
                dim = out
                x = h
                print(x.shape)

            with tf.variable_scope('conv2'):
                out = dim * 2
                w = weight_variable([3, 3, dim, out])
                b = bias_variable([out])
                h = max_pool_2x2(tf.nn.relu(conv2d(x, w) + b))
                dim = out
                x = h
                print(x.shape)

            with tf.variable_scope('conv3'):
                out = dim * 2
                w = weight_variable([3, 3, dim, out])
                b = bias_variable([out])
                h = max_pool_2x2(tf.nn.relu(conv2d(x, w) + b))
                dim = out
                x = h
                dim = 128
                x = tf.reshape(x, [-1, dim])
                self.convolution_embedding = x
                print(x.shape)

            with tf.variable_scope('read_out'):
                non_rotated_embedding = self.no_embedding_classifier.convolution_embedding
                embedding_dist = compute_euclidean_distances(non_rotated_embedding, self.convolution_embedding)
                embedding_dist = embedding_dist / tf.norm(non_rotated_embedding, axis=1)
                print("embedding shape: {}".format(embedding_dist.shape))
                self.loss = tf.reduce_mean(embedding_dist)

    def train(self, data_generator, batch_size=50, iterations=100, log_freq=5, keep_prob=1.0):
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             "rotated_embedding")
        train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss, var_list=train_vars)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            run_name = './train/run_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            print(run_name)

            for i in range(iterations):
                batch_x, batch_y_ = data_generator.train(batch_size)

                if i % log_freq == 0:
                    loss, = sess.run([self.loss],
                                         feed_dict={self.x: batch_x, self.keep_prob: 1.0})
                    v_batch_x, v_batch_y_ = data_generator.validation()
                    v_loss,  = sess.run([self.loss],
                                             feed_dict={self.x: v_batch_x, self.keep_prob: 1.0})
                    v_acc = "-1"
                    print(
                        'iteration {}, training loss {}, validation loss {}, acc {}'.format(i, loss, v_loss, v_acc))

                train_step.run(feed_dict={self.x: batch_x, self.keep_prob: keep_prob})

            t_batch_x, t_batch_y_ = data_generator.test(augment=False)
            t_loss, = sess.run([self.loss],
                                     feed_dict={self.x: t_batch_x, self.keep_prob: 1.0})
            print('test loss {}, test accuracy'.format(t_loss))

if __name__ == "__main__":
    classifier = NoEmbeddingClassifier()
    embeddor = RotatedEmbeddingClassifier(classifier)
    data_generator = RotatedMNISTDataGenerator(augment=False)
    classifier.train(data_generator=data_generator, batch_size=200, iterations=100)

    data_generator = RotatedMNISTDataGenerator(augment=False)
    embeddor.train(data_generator=data_generator, batch_size=200, iterations=10)


