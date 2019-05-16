import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.contrib.layers import flatten
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class LeNet5(object):

    def construct_net(self, x):
        # Hyperparameters
        mu = 0
        sigma = 0.1
        layer_depth = {
            'layer_1': 6,
            'layer_2': 16,
            'layer_3': 120,
            'layer_f1': 84
        }

        # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        # TODO: Activation.
        conv1 = tf.nn.relu(conv1)

        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
        pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # TODO: Layer 2: Convolutional. Output = 10x10x16.
        conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        # TODO: Activation.
        conv2 = tf.nn.relu(conv2)

        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # TODO: Flatten. Input = 5x5x16. Output = 400.
        fc1 = flatten(pool_2)

        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc1, fc1_w) + fc1_b

        # TODO: Activation.
        fc1 = tf.nn.relu(fc1)

        # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(84))
        fc2 = tf.matmul(fc1, fc2_w) + fc2_b
        # TODO: Activation.
        fc2 = tf.nn.relu(fc2)

        # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
        fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
        fc3_b = tf.Variable(tf.zeros(10))
        logits = tf.matmul(fc2, fc3_w) + fc3_b

        return logits

    def model(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, num_epochs=10, learning_rate=0.001,
              minibatch_size=64,
              print_cost=True):

        # tf.reset_default_graph()

        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None, 10))

        logits = self.construct_net(x)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # 计算正确率
        predict_op = tf.argmax(logits, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy)

        costs = []
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):

                num_examples = len(X_train)
                minibatch_cost = 0.0

                for i in range(0, num_examples, minibatch_size):
                    batch_x, batch_y = X_train[i:i + minibatch_size], Y_train[i:i + minibatch_size]

                    _, temp_cost = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                    minibatch_cost += temp_cost / int(num_examples / minibatch_size)

                if print_cost == True and epoch % 5 == 0:
                    print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                    # 计算开发集正确率
                    validation_accuracy = self.evaluate(x, y, X_validation, Y_validation, minibatch_size, accuracy)
                    print("Validation Accuracy", validation_accuracy)

                if print_cost == True and epoch % 1 == 0:
                    costs.append(minibatch_cost)

            saver.save(sess, "./ckpt/lenet_model")

            plt.plot(np.squeeze(costs))
            plt.ylabel("cost")
            plt.xlabel("iterations (per tens)")
            plt.title("learning_rate=" + str(learning_rate))
            plt.show()

        # 计算测试集正确率
        with tf.Session() as sess1:
            saver.restore(sess1, './ckpt/lenet_model')
            test_accuracy = self.evaluate(x, y, X_test, Y_test, minibatch_size, accuracy)
            print("Test Accuracy", test_accuracy)

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


if __name__ == '__main__':
    mnist = input_data.read_data_sets("./minist_data/input_data/", one_hot=True)

    # 获取train/dev/test set
    X_train, Y_train = mnist.train.images, mnist.train.labels
    X_validation, Y_validation = mnist.validation.images, mnist.validation.labels
    X_test, Y_test = mnist.test.images, mnist.test.labels
    print(X_train.shape)

    assert (len(X_train) == len(Y_train))
    assert (len(X_validation) == len(Y_validation))
    assert (len(X_test) == len(Y_test))

    X_train = np.reshape(X_train, [-1, 28, 28, 1])
    X_validation = np.reshape(X_validation, [-1, 28, 28, 1])
    X_test = np.reshape(X_test, [-1, 28, 28, 1])

    # Pad images with 0s
    X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    X_train, Y_train = shuffle(X_train, Y_train)

    nn = LeNet5()
    nn.model(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
