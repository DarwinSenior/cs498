import mnist_helper as H
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':

    FLAGS = H.flags().FLAGS
    # NN specification
    x, y_, keep_prob = H.input_layer()
    hidden1 = H.nn_layer(x, 784, 500, 'layer1')
    dropped = tf.nn.dropout(hidden1, keep_prob)
    y = H.nn_layer(dropped, 500, 10, 'layer2', act=tf.nn.softmax)
    train_step, accuracy = H.loss(y_, y, FLAGS.learning_rate)


    sess = tf.InteractiveSession()
    mnist = input_data.read_data_sets(
        FLAGS.data_dir, one_hot=True, fake_data=False)
    tf.initialize_all_variables().run()

    # summary specification
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(
        FLAGS.summaries_dir + '-train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '-test')

    # read inputs

    # train
    for i in range(FLAGS.max_steps):
        if i % 100 == 0:
            dropout = FLAGS.dropout
            feed_dict = H.feed_dict(mnist.test, dropout, x, y_, keep_prob, isTrain=False)
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict)
            print('Accuracy at step %i: %i'%(i, acc))
        else:
            dropout = FLAGS.dropout
            feed_dict = H.feed_dict(mnist.train, dropout, x, y_, keep_prob, isTrain=True)
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
            train_writer.add_summary(summary, i)

