import tensorflow as tf
from cifar10 import _activation_summary, _variable_with_weight_decay, _variable_on_cpu, NUM_CLASSES, FLAGS


class NeuralNet(object):

    def __init__(self, images):
        self.input = images

    def conv(self, name, shape, stddev, wd=0.0, bias_init=0.0, padding='SAME', activation=tf.nn.relu):
        with tf.variable_scope(name):
            kernel = _variable_with_weight_decay('weights', shape=shape,
                                                 stddev=stddev, wd=wd)
            conv = tf.nn.conv2d(self.input, kernel, [
                                1, 1, 1, 1], padding=padding)
            biases = _variable_on_cpu(
                'biases', [shape[-1]], tf.constant_initializer(bias_init))
            bias = tf.nn.bias_add(conv, biases)
            conv = activation(bias, name=name)
            _activation_summary(conv)
            self.input = conv
            return self

    def pool(self, name, ksize, stride, padding='SAME'):
        self.input = tf.nn.max_pool(self.input,
                                    ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                                    padding=padding, name=name)
        return self

    def norm(self, name, sz=4, bias=1.0, alpha=0.001 / 0.9, beta=0.75):
        self.input = tf.nn.lrn(self.input, sz, bias=bias,
                               alpha=alpha, beta=beta, name=name)
        return self

    def reshape(self, size):
        self.input = tf.reshape(self.input, size)
        return self

    def full(self, name, shape, stddev, wd=0.0, bias_init=0.0, activation=tf.nn.relu):
        with tf.variable_scope(name):
            weights = _variable_with_weight_decay(
                'weights', shape=shape, stddev=stddev, wd=wd)
            biases = _variable_on_cpu(
                'biases', [shape[-1]], tf.constant_initializer(bias_init))
            local = tf.nn.relu(
                tf.matmul(self.input, weights) + biases, name=name)
            _activation_summary(local)
            self.input = local
        return self


def default_nn(images, batch_size, NUM_CLASSES):
    nn = (
        NeuralNet(images)
        .conv('conv1', [5, 5, 3, 64], stddev=1e-4, bias_init=0)
        .pool('pool1', ksize=3, stride=2)
        .norm('norm1')
        .conv('conv2', [5, 5, 64, 64], stddev=1e-4, bias_init=0.1)
        .norm('norm2')
        .pool('pool2', ksize=3, stride=2)
        .reshape([batch_size, -1])
        .full('local3', [2304, 384], stddev=0.04, wd=0.004)
        .full('local4', [384, 192], stddev=0.04, wd=0.004)
        .full('softmax', [192, NUM_CLASSES], stddev=1 / 192, wd=0.0, activation=lambda x: x)
    )
    return nn.input


def inference(images):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 32],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 32, 32],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix
        # multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) +
                            biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) +
                            biases, name=scope.name)
        _activation_summary(local4)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(
            tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear

def inference2(images):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    with tf.variable_scope('conv12') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv12 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv12)
    # pool1

    pool1 = tf.nn.max_pool(conv12, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # conv2
    with tf.variable_scope('conv22') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv22 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv22)

    # norm2
    norm2 = tf.nn.lrn(conv22, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix
        # multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 6*64],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [64*6], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) +
                            biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[64*6, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) +
                            biases, name=scope.name)
        _activation_summary(local4)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(
            tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear
