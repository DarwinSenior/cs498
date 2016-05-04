from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs2', 'Summaries directory')


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


def max_pool_2x2(x, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)


def conv_layer(shape, input, activation, padding):
    w_conv = weight_variable(shape)
    b_conv = bias_variable([shape[-1]])
    return activation(conv2d(input, w_conv, padding) + b_conv)


def full_layer(shape, input, activation):
    w = weight_variable(shape)
    b = bias_variable([shape[-1]])
    return activation(tf.matmul(input, w) + b)


class NeuralNet(object):

    def __init__(self, x, y, debug=False):
        self.input = x
        self.y = y
        self.debug = debug

    def reshape(self, shape):
        if (self.debug):
            print(self.input.get_shape())
        self.input = tf.reshape(self.input, shape)
        return self

    def conv_layer(self, shape, activation=tf.nn.relu, padding='VALID'):
        if (self.debug):
            print(self.input.get_shape())
        self.input = conv_layer(shape, self.input, activation, padding=padding)
        return self

    def max_pool_2x2(self, padding='SAME'):
        if (self.debug):
            print(self.input.get_shape())
        self.input = max_pool_2x2(self.input, padding=padding)
        return self

    def dropout(self, keep_prob):
        if (self.debug):
            print(self.input.get_shape())
        self.input = tf.nn.dropout(self.input, keep_prob)
        return self

    def full_layer(self, shape, activation=tf.nn.relu):
        if (self.debug):
            print(self.input.get_shape())
        self.input = full_layer(shape, self.input, activation)
        return self

    def cross_entropy(self):
        return tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.input), reduction_indices=[1]))

    def accuracy(self):
        correct_prediction = tf.equal(tf.arg_max(
            self.input, 1), tf.arg_max(self.y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', acc)
        return acc


def cnn_before():
    cnn = NeuralNet(x, y_, debug=True)
    (cnn
     .reshape([-1, 28, 28, 1])
     .conv_layer([5, 5, 1, 32], padding='SAME')
     .max_pool_2x2()
     .conv_layer([5, 5, 32, 64], padding='SAME')
     .max_pool_2x2()
     .reshape([-1, 7 * 7 * 64])
     .full_layer([7 * 7 * 64, 1024])
     .dropout(keep_prob)
     .full_layer([1024, 10], activation=tf.nn.softmax))
    return cnn


def cnn_smaller():
    """
    comparing to the other one, this architecture
    uses smaller paddings, and the accuracy is slightly increased
    """
    cnn = NeuralNet(x, y_, debug=True)
    (cnn
     .reshape([-1, 28, 28, 1])
     .conv_layer([3, 3, 1, 32], padding='SAME')
     .max_pool_2x2()
     .conv_layer([3, 3, 32, 64], padding='SAME')
     .max_pool_2x2()
     .conv_layer([3, 3, 64, 64], padding='VALID')
     .conv_layer([3, 3, 64, 64], padding='VALID')
     .reshape([-1, 3 * 3 * 64])
     .full_layer([3 * 3 * 64, 1024])
     .dropout(keep_prob)
     .full_layer([1024, 10], activation=tf.nn.softmax))
    return cnn


def cnn_use_padding():
    """
    this network shrink the padding instead of using the full padding
    """
    cnn = NeuralNet(x, y_, debug=True)
    (cnn
     .reshape([-1, 28, 28, 1])
     .conv_layer([3, 3, 1, 32], padding='VALID')
     .max_pool_2x2()
     .conv_layer([4, 4, 32, 64], padding='VALID')
     .max_pool_2x2()
     .reshape([-1, 5 * 5 * 64])
     .full_layer([5 * 5 * 64, 1024])
     .dropout(keep_prob)
     .full_layer([1024, 10], activation=tf.nn.softmax))
    return cnn


def cnn_enlarged():
    """
    this network increased the depth of the layers
    """
    cnn = NeuralNet(x, y_, debug=True)
    (cnn
     .reshape([-1, 28, 28, 1])
     .conv_layer([3, 3, 1, 64], padding='VALID')
     .max_pool_2x2()
     .conv_layer([4, 4, 64, 128], padding='VALID')
     .max_pool_2x2()
     .reshape([-1, 5 * 5 * 128])
     .full_layer([5 * 5 * 128, 1024])
     .dropout(keep_prob)
     .full_layer([1024, 10], activation=tf.nn.softmax))
    return cnn


def cnn_with_another():
    """
    it is tried to reduce the convolutional layers
    with not max pool but with conv layer
    """
    cnn = NeuralNet(x, y_, debug=True)
    (cnn
     .reshape([-1, 28, 28, 1])
     .conv_layer([6, 6, 1, 8], padding='VALID')
     .conv_layer([4, 4, 8, 16], padding='VALID')
     .conv_layer([3, 3, 16, 32], padding='VALID')
     .max_pool_2x2()
     .conv_layer([3, 3, 32, 64], padding='VALID')
     .conv_layer([3, 3, 64, 128], padding='VALID')
     .reshape([-1, 5 * 5 * 128])
     .full_layer([5 * 5 * 128, 1024])
     .dropout(keep_prob)
     .full_layer([1024, 10], activation=tf.nn.softmax))

keep_prob = tf.placeholder(tf.float32)

cnn = NeuralNet(x, y_, debug=True)
(cnn
 .reshape([-1, 28, 28, 1])
 .conv_layer([3, 3, 1, 32], padding='SAME')
 .max_pool_2x2()
 .conv_layer([3, 3, 32, 64], padding='SAME')
 .max_pool_2x2()
 .conv_layer([2, 2, 64, 128], padding='VALID')
 .max_pool_2x2()
 .reshape([-1, 3 * 3 * 128])
 .full_layer([3 * 3 * 128, 1024])
 .dropout(keep_prob)
 .full_layer([1024, 10], activation=tf.nn.softmax))

cross_entropy = cnn.cross_entropy()
accuracy = cnn.accuracy()

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver = tf.train.Saver()

# Merge all the summaries and write them out to /tmp/mnist_logs (by
# default)
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(
    FLAGS.summaries_dir + '/train', sess.graph)
test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
tf.initialize_all_variables().run()


sess.run(tf.initialize_all_variables())
for i in range(50000):
    xs, ys = mnist.train.next_batch(50)
    summary, train_accuracy = sess.run([merged, train_step],
                       feed_dict={x: xs,
                                  y_: ys,
                                  keep_prob: 0.3})
    train_writer.add_summary(summary, i)
    if (i + 1) % 100 == 0:
        [summary, test_accuracy] = sess.run([merged, accuracy],
                           feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels,
                                      keep_prob: 1})
        test_writer.add_summary(summary, i)
        print("train accuracy at step %i is %s" % (i, train_accuracy))
        print("test accuracy at step %i is %s" % (i, test_accuracy))
        # print("saved at %s" % save_path)
