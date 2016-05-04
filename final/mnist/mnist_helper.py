import tensorflow as tf

def flags(fake_data=False, max_steps=10000, learning_rate=0.001, dropout=0.9):
    flags = tf.app.flags
    flags.DEFINE_boolean('fake_data', fake_data, 'If true, uses fake data '
                         'for unit testing.')
    flags.DEFINE_integer('max_steps', max_steps, 'Number of steps to run trainer.')
    flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate.')
    flags.DEFINE_float('dropout', dropout, 'Keep probability for training dropout.')
    flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
    flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')
    return flags

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def input_layer(input_size=28, img_dim=1, num_classes=10):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, input_size*input_size*img_dim])
        image_shaped_input = tf.reshape(x, [-1, input_size, input_size, img_dim])
        tf.image_summary('input', image_shaped_input, num_classes)
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)
    return x, y_, keep_prob

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read, and
    adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the
    # graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.histogram_summary(
                layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, 'activation')
        tf.histogram_summary(layer_name + '/activations', activations)
        return activations

def cross_entropy(y_, y):
    diff = y_ * tf.log(y)
    return -tf.reduce_mean(diff)

def mostlikely(y, y_):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)
    return accuracy

def loss(y_, y, learning_rate, lossfunc=cross_entropy, optimiser=tf.train.AdamOptimizer, accuracy=mostlikely):
    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            loss = lossfunc(y_, y);
            tf.scalar_summary('loss', loss)

    with tf.name_scope('train'):
        train_step = optimiser(learning_rate).minimize(loss)

    with tf.name_scope('accuracy'):
        acc = accuracy(y, y_)
        tf.scalar_summary('accuracy', accuracy(y, y_))
    return train_step, acc

def feed_dict(data, dropout, x, y_, keep_prob, isTrain=True):
    """
    For there, if istraining = true, data = mnist.train else data = mnist.test
    """
    if isTrain:
        xs, ys = data.next_batch(100, fake_data=False)
        k = FLAGS.dropout
    else:
        xs, ys = data.images, data.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

