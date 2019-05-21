import tensorflow as tf


def evaluate(logits, y):
    # 计算正确率
    predict_op = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy)

    return accuracy


