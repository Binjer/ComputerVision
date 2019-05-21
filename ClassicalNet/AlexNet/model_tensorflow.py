# 用tensorflow搭建AlexNet网络

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def interface(images):
    parameters = []
    # # test
    # with tf.variable_scope("test"):
    #     conv1_w = tf.Variable(tf.truncated_normal(shape=[11, 11, 3, 2], mean=0, stddev=0.01))
    #     conv1_b = tf.Variable(tf.zeros(2))
    #     conv1 = tf.nn.conv2d(images, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    #
    #     print_activations(conv1)
    #     parameters += [conv1_w, conv1_b]
    #
    # pool1 = tf.nn.max_pool(conv1,
    #                        ksize=[1, 5, 5, 1],
    #                        strides=[1, 2, 2, 1],
    #                        padding='VALID',
    #                        name='pool1')
    # print_activations(pool1)
    #
    # with tf.variable_scope("fc1"):
    #     fc1 = flatten(pool1)
    #     fc1_w = tf.Variable(tf.truncated_normal([162, 10],
    #                                             dtype=tf.float32,
    #                                             stddev=1e-1))
    #     fc1_b = tf.Variable(tf.zeros(10))
    #
    #     fc1 = tf.matmul(fc1, fc1_w) + fc1_b
    #     # fc1 = tf.nn.relu(fc1)
    #     parameters += [fc1_w, fc1_b]
    #
    # return parameters, fc1

    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    # lrn1
    # TODO(shlens, jiayq): Add a GPU version of local response normalization.

    # pool1
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')
    print_activations(pool1)

    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    # pool2
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    print_activations(pool5)

    # fc1
    with tf.name_scope('fc1') as scope:
        fc1 = flatten(pool5)
        fc1_w = tf.Variable(tf.truncated_normal([9216, 4096],
                                                dtype=tf.float32,
                                                stddev=1e-1), name="weights")
        fc1_b = tf.Variable(tf.zeros(4096), name="bias")

        fc1 = tf.matmul(fc1, fc1_w) + fc1_b
        fc1 = tf.nn.relu(fc1)
        parameters += [fc1_w, fc1_b]

    # fc2
    with tf.name_scope('fc2') as scope:
        fc2_w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                dtype=tf.float32,
                                                stddev=1e-1), name="weights")
        fc2_b = tf.Variable(tf.zeros(4096), name="bias")

        fc2 = tf.matmul(fc1, fc2_w) + fc2_b
        fc2 = tf.nn.relu(fc2)
        parameters += [fc2_w, fc2_b]

    # output
    with tf.name_scope('output') as scope:
        fc3_w = tf.Variable(tf.truncated_normal([4096, 10],
                                                dtype=tf.float32,
                                                stddev=1e-1), name="weights")
        fc3_b = tf.Variable(tf.zeros(10), name="bias")

        fc3_logits = tf.matmul(fc2, fc3_w) + fc3_b
        parameters += [fc3_w, fc3_b]
        print(fc3_logits)

    return parameters, fc3_logits


def compute_cost(logits, labels):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


def evaluate(self, x, y, X_data, Y_data, minibatch_size, accuracy_op):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()

    total_accuracy = 0.0
    for i in range(0, num_examples, minibatch_size):
        batch_x, batch_y = X_data[i:i + minibatch_size], Y_data[i:i + minibatch_size]
        accuracy = sess.run(accuracy_op, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples
