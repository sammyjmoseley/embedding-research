from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

sess = tf.InteractiveSession()

x_data = sess.run(tf.random_uniform([10, 2]))
m = tf.constant([[1.0], [2.0]])
y_data = tf.matmul(x_data, m).eval()
train_x = tf.contrib.data.Dataset.from_tensors(x_data)
train_y = tf.contrib.data.Dataset.from_tensors(y_data)
# mnist = tf.contrib.data.Dataset.zip((train_x, train_y))



x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x, W) + b

cross_entropy = tf.nn.l2_loss(tf.subtract(y_, y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


for _ in range(10000):
  # batch = mnist.next_batch(100)
  # train_step.run(feed_dict={x: train_x, y_: train_y})
  sess.run(train_step, feed_dict={x: x_data, y_: y_data})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(W.eval())
