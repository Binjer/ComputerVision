from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np

nb_classes = 10


def load_dataset():
    # 读取数据集
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # 将类别转化为向量
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    print(Y_train.shape)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = load_dataset()

# print(type(X_train))

# X_train = tf.image.resize_images(X_train, [227, 227], method=0)
# print(X_train.shape)
# print(X_train[3000])
# print(type(X_train[3000]))
#
# with tf.Session() as sess:
#     X_train = sess.run(X_train)

# plt.imshow(X_train[3000])
# plt.show()

# 图像放缩, 需要较大的内存
# num_examples = X_train.shape[0]
#
# first_batch = X_train[0:500, :, :, :]
# temp_resize = np.zeros((500, 227, 227, 3))
# for i in range(500):
#     temp = first_batch[i]
#     temp_resize[i] = cv2.resize(temp, (227, 227), interpolation=cv2.INTER_CUBIC)
#
# # print(temp_resize.shape)
#
# # second_batch = X_train[500:1000, :, :, :]
# # temp_resize1 = np.zeros((500, 227, 227, 3))
# # for i in range(500):
# #     temp = second_batch[i]
# #     temp_resize1[i] = cv2.resize(temp, (227, 227), interpolation=cv2.INTER_CUBIC)
# #
# # tmp = np.concatenate((temp_resize, temp_resize1), axis=0)
#
# X_train_resize = None
# for i in range(1, 5):
#     batch_x = X_train[(i * 500):((i + 1) * 500), :, :, :]
#
#     tmp1 = np.zeros((500, 227, 227, 3))
#     for j in range(500):
#         tmp1[j] = cv2.resize(batch_x[j], (227, 227), interpolation=cv2.INTER_CUBIC)
#
#     X_train_resize = np.concatenate((temp_resize, tmp1))

# print(temp_resize1[3])
# plt.imshow(X_train_resize[700])
# plt.show()
