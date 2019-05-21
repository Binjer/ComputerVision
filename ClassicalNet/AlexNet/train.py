import tensorflow as tf
from input_cifar10 import X_train, Y_train, X_test, Y_test
from model_tensorflow import *
from eval import *
import numpy as np


def model(X_train, Y_train, X_test, Y_test, num_epochs=10, learning_rate=0.001, minibatch_size=64,
          print_cost=True):
    x = tf.placeholder(tf.float32, (None, 32, 32, 3), name="X")
    y = tf.placeholder(tf.float32, (None, 10), name="Y")

    x_resize = tf.image.resize_images(x, (227, 227), method=0)

    parameters, logits = interface(x_resize)
    cost = compute_cost(logits, y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    accuracy = evaluate(logits, y)

    saver = tf.train.Saver()
    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # with tf.device("/cpu:0"):
        sess.run(init)

        for epoch in range(num_epochs):

            num_examples = len(X_train)
            minibatch_cost = 0.0

            for i in range(0, num_examples, minibatch_size):
                batch_x, batch_y = X_train[i:i + minibatch_size], Y_train[i:i + minibatch_size]

                _, temp_cost = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                minibatch_cost += temp_cost / int(num_examples / minibatch_size)

            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))

            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        saver.save(sess, "./ckpt/alexnet_model")

    # 计算训练集正确率
    num_train = len(X_train)

    with tf.Session() as sess:
        saver.restore(sess, './ckpt/alexnet_model')

        train_total_accuracy = 0.0
        for i in range(0, num_train, minibatch_size):
            batch_x_train, batch_y_train = X_train[i:i + minibatch_size], Y_train[i:i + minibatch_size]
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_x_train, y: batch_y_train})
            train_total_accuracy += train_accuracy * len(batch_x_train)

        train_total_accuracy /= num_train
        print("Train Accuracy", train_total_accuracy)

    # 计算测试集正确率
    num_test = len(X_test)

    with tf.Session() as sess:
        saver.restore(sess, './ckpt/alexnet_model')

        test_total_accuracy = 0.0
        for i in range(0, num_test, minibatch_size):
            batch_x, batch_y = X_test[i:i + minibatch_size], Y_test[i:i + minibatch_size]
            test_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            test_total_accuracy += test_accuracy * len(batch_x)

        test_total_accuracy /= num_test
        print("Test Accuracy", test_total_accuracy)

    return costs, train_total_accuracy, test_total_accuracy


if __name__ == '__main__':
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    # X_train = np.pad(X_train, ((0, 0), (98, 97), (98, 97), (0, 0)), "constant")
    print(X_train.shape)

    X_test = X_test[:100]
    Y_test = Y_test[:100]

    costs, train_accuracy, test_accuracy = model(X_train, Y_train, X_test, Y_test)
