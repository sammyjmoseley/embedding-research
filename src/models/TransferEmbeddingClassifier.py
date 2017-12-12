# Full network trained end-to-end on classification
# (No embedding component at all)

from models import Embedding
from models import Classifier
import tensorflow as tf
import numpy as np
from data_generators.TransferEmbeddingDataGenerator import RotatedMNISTDataGenerator
from datetime import datetime

dropout = 0.8


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.get_variable("weights", dtype=tf.float32, initializer=initial)


def identity_weight_variable(var):
    return tf.get_variable("weights", dtype=tf.float32, initializer=var)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape, dtype=tf.float32)
    return tf.get_variable("biases", dtype=tf.float32, initializer=initial)


def identity_bias_variable(var):
    return tf.get_variable("biases", dtype=tf.float32, initializer=var)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def compute_euclidean_distances(x, y, w=None):
    d = tf.square(tf.subtract(x, y))
    d = tf.sqrt(tf.reduce_sum(d, axis=1))
    return d


def create_fc_layer(this, name, x, dim, out, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        x = tf.nn.dropout(x, keep_prob=this.keep_prob)

        w = weight_variable([dim, out])
        b = bias_variable([out])
        h = tf.nn.relu(tf.matmul(x, w) + b)
        x = h
        return x

class NoEmbeddingClassifier:
    def __init__(self):

        self.layers = [("fc1", (128, 10))]

        with tf.variable_scope('classifier'):
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

            x = self.convolution_embedding

            for layer in self.layers:
                name, (dim, out) = layer
                x = create_fc_layer(self, name, x, dim, out)
                dim = out

            with tf.variable_scope('softmax'):
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

            self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.class_loss)

    def train(self, data_generator, sess, batch_size=50, iterations=100, log_freq=5, keep_prob=1.0):
        run_name = './train/run_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print(run_name)

        for i in range(iterations):
            batch_x, batch_y_ = data_generator.train(batch_size)

            if i % log_freq == 0:
                loss, = sess.run([self.class_loss],
                                     feed_dict={self.x: batch_x, self.y_: batch_y_, self.keep_prob: keep_prob})
                v_batch_x, v_batch_y_ = data_generator.validation()
                v_loss, v_acc = sess.run([self.class_loss, self.accuracy],
                                         feed_dict={self.x: v_batch_x, self.y_: v_batch_y_, self.keep_prob: 1.0})
                print(
                    'iteration {}, training loss {}, validation loss {}, acc {}'.format(i, loss, v_loss, v_acc))

            self.train_step.run(feed_dict={self.x: batch_x, self.y_: batch_y_, self.keep_prob: keep_prob})

        (t_batch_x, t_batch_x_), t_batch_y_ = data_generator.test(augment=True)

        t_loss, acc = sess.run([self.class_loss, self.accuracy],
                               feed_dict={self.x: t_batch_x_,
                                          self.y_: t_batch_y_,
                                          self.keep_prob: 1.0,
                                          })
        print('test loss no augment {}, test accuracy {}'.format(t_loss, acc))

        t_loss, acc = sess.run([self.class_loss, self.accuracy],
                               feed_dict={self.x: t_batch_x,
                                          self.y_: t_batch_y_,
                                          self.keep_prob: 1.0})
        print('test loss augment {}, test accuracy {}'.format(t_loss, acc))

    def convolution_embed(self, sess, x):
        embedding, = sess.run([self.convolution_embedding],
                         feed_dict={self.x: x, self.keep_prob: 1.0})
        return embedding


class RotatedEmbeddingClassifier:
    def __init__(self, no_embedding_classifier):
        self.no_embedding_classifier = no_embedding_classifier


        with tf.variable_scope('rotated_embedding'):
            # Input and label placeholders
            with tf.variable_scope('input'):
                self.x_ = no_embedding_classifier.x
                self.x = tf.placeholder(tf.float32, shape=self.x_.shape, name="x")
                self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            dim = 1
            scale = 10
            init_dim = 2*scale
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

            with tf.variable_scope('conv3'):
                out = dim * 2
                w = weight_variable([3, 3, dim, out])
                b = bias_variable([out])
                h = max_pool_2x2(tf.nn.relu(conv2d(x, w) + b))
                dim = out
                x = h
                dim = 128*scale
                x = tf.reshape(x, [-1, dim])

            with tf.variable_scope('conv_fc1'):
                out = dim
                w = weight_variable([dim, out])
                b = bias_variable([out])
                x = tf.nn.relu(tf.matmul(x, w) + b)
                dim = out

            with tf.variable_scope('conv_fc2'):
                out = dim
                w = weight_variable([dim, out])
                b = bias_variable([out])
                x = tf.nn.relu(tf.matmul(x, w) + b)
                dim = out

            with tf.variable_scope('read_out'):
                out = 128
                w = weight_variable([dim, out])
                b = bias_variable([out])
                x = tf.matmul(x, w) + b
                dim = out
                self.convolution_embedding = x
                non_rotated_embedding = self.no_embedding_classifier.convolution_embedding
                embedding_dist = compute_euclidean_distances(non_rotated_embedding, self.convolution_embedding)
                self.embedding_loss = tf.reduce_mean(tf.square(embedding_dist))

        with tf.variable_scope('classifier'):
            for layer in self.no_embedding_classifier.layers:
                name, (dim, out) = layer
                x = create_fc_layer(self, name, x, dim, out, reuse=True)
                dim = out

            with tf.variable_scope('softmax', reuse=True):
                out = 10
                x = tf.nn.dropout(x, keep_prob=self.keep_prob)
                w = weight_variable([dim, out])
                b = bias_variable([out])
                self.before_softmax = tf.matmul(x, w) + b
                print(self.before_softmax.shape)
                self.y = tf.nn.softmax(self.before_softmax, dim=1)

            with tf.variable_scope('class_loss', reuse=True):
                print(self.y_.shape)
                print(self.before_softmax.shape)
                self.class_loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.variable_scope('rotated_embedding'):
            train_vars_conv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                "rotated_embedding")
            train_vars_fc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                "classifier")

            self.train_step_conv = tf.train.AdamOptimizer(1e-3, name="AdamConv").minimize(self.embedding_loss,
                                                                         var_list=train_vars_conv)
            self.train_step_fc = tf.train.AdamOptimizer(1e-3, name="AdamFC").minimize(self.class_loss,
                                                                   var_list=train_vars_fc)

    def train_convolution(self, data_generator, sess, batch_size=50, iterations=100, log_freq=5, keep_prob=1.0):
        run_name = './train/run_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print(run_name)

        for i in range(iterations):
            (batch_x, batch_x_), batch_y_ = data_generator.train(batch_size)

            if i % log_freq == 0:
                loss, = sess.run([self.embedding_loss],
                                     feed_dict={self.x: batch_x,
                                                self.x_: batch_x_,
                                                self.y_: batch_y_,
                                                self.keep_prob: keep_prob})
                (v_batch_x, v_batch_x_), v_batch_y_ = data_generator.validation()
                v_loss,  = sess.run([self.embedding_loss],
                                         feed_dict={self.x: v_batch_x,
                                                    self.x_: v_batch_x_,
                                                    self.y_: v_batch_y_,
                                                    self.keep_prob: 1.0})
                v_acc = "-1"
                print(
                    'iteration {}, training loss {}, validation loss {}, acc {}'.format(i, loss, v_loss, v_acc))

            self.train_step_conv.run(feed_dict={self.x: batch_x,
                                                self.x_: batch_x,
                                                self.y_: batch_y_,
                                                self.keep_prob: keep_prob})

        t_batch_x, t_batch_y_ = data_generator.test(augment=False)
        t_loss, acc = sess.run([self.embedding_loss, self.accuracy],
                                 feed_dict={self.x: t_batch_x,
                                            self.x_: t_batch_x,
                                            self.y_: t_batch_y_,
                                            self.keep_prob: 1.0,
                                            })
        print('test loss no augment {}, test accuracy {}'.format(t_loss, acc))

        (t_batch_x, t_batch_x_), t_batch_y_ = data_generator.test(augment=True)
        t_loss, acc = sess.run([self.embedding_loss, self.accuracy],
                           feed_dict={self.x: t_batch_x,
                                      self.x_: t_batch_x_,
                                      self.y_: t_batch_y_,
                                      self.keep_prob: 1.0})
        print('test loss augment {}, test accuracy {}'.format(t_loss, acc))

    def train_fc(self, data_generator, sess, batch_size=50, iterations=100, log_freq=5, keep_prob=1.0):
        run_name = './train/run_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print(run_name)

        for i in range(iterations):
            batch_x, batch_y_ = data_generator.train(batch_size)

            if i % log_freq == 0:
                loss, = sess.run([self.class_loss],
                                     feed_dict={self.x: batch_x, self.y_: batch_y_, self.keep_prob: keep_prob})
                v_batch_x, v_batch_y_ = data_generator.validation()
                v_loss, v_acc = sess.run([self.class_loss, self.accuracy],
                                         feed_dict={self.x: v_batch_x, self.y_: v_batch_y_, self.keep_prob: 1.0})
                print(
                    'iteration {}, training loss {}, validation loss {}, acc {}'.format(i, loss, v_loss, v_acc))

            self.train_step_fc.run(feed_dict={self.x: batch_x, self.y_: batch_y_, self.keep_prob: keep_prob})

        t_batch_x, t_batch_y_ = data_generator.test(augment=False)
        t_loss, acc = sess.run([self.class_loss, self.accuracy],
                               feed_dict={self.x: t_batch_x,
                                          self.x_: t_batch_x,
                                          self.y_: t_batch_y_,
                                          self.keep_prob: 1.0,
                                          })
        print('test loss no augment {}, test accuracy {}'.format(t_loss, acc))

        (t_batch_x, t_batch_x_), t_batch_y_ = data_generator.test(augment=True)
        t_loss, acc = sess.run([self.class_loss, self.accuracy],
                               feed_dict={self.x: t_batch_x,
                                          self.x_: t_batch_x_,
                                          self.y_: t_batch_y_,
                                          self.keep_prob: 1.0})
        print('test loss augment {}, test accuracy {}'.format(t_loss, acc))

if __name__ == "__main__":
    classifier = NoEmbeddingClassifier()
    embeddor = RotatedEmbeddingClassifier(classifier)
    data_generator = RotatedMNISTDataGenerator(ang_range=(-30, 30))

    with tf.Session() as sess:
        classifier_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope="classifier")
        rotated_embedding = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope="rotated_embedding")
        sess.run(tf.variables_initializer(classifier_vars))
        classifier.train(data_generator=data_generator,
                         sess=sess,
                         batch_size=200,
                         iterations=3000,
                         keep_prob=dropout,
                         log_freq=100)
        # layers = ["conv1", "conv2", "conv3"]
        # for layer in layers:
        #
        #     var1 = tf.get_default_graph().get_tensor_by_name("rotated_embedding/{}/weights:0".format(layer))
        #     var2 = tf.get_default_graph().get_tensor_by_name("classifier/{}/weights:0".format(layer))
        #     tf.assign(var1, var2, name="rotated_embedding/{}/weights".format(layer))
        #
        #     var1 = tf.get_default_graph().get_tensor_by_name("rotated_embedding/{}/biases:0".format(layer))
        #     var2 = tf.get_default_graph().get_tensor_by_name("classifier/{}/biases:0".format(layer))
        #     tf.assign(var1, var2, name="rotated_embedding/{}/biases".format(layer))
        #
        #     # tf.assign(tf.get_variable("rotated_embedding/{}/weights".format(layer)),
        #     #           tf.get_variable("classifier/{}/weights".format(layer)))
        sess.run(tf.variables_initializer(rotated_embedding))

        data_generator_augmented = RotatedMNISTDataGenerator(ang_range=(-30, 30), augment=True)
        embeddor.train_convolution(data_generator=data_generator_augmented,
                                   sess=sess, batch_size=200,
                                   iterations=2000,
                                   log_freq=100,
                                   keep_prob=dropout)
        embeddor.train_fc(data_generator=data_generator,
                          sess=sess, batch_size=200,
                          iterations=1000,
                          log_freq=100,
                          keep_prob=dropout)



