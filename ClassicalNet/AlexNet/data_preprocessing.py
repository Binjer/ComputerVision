# 如何从二进制cifar10文件中读取图片数据
# 如何把二进制cifar10文件转换为tfrecords格式

import tensorflow as tf
import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 定义cifar的数据等命令行参数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("cifar_dir", "./data/cifar-10-batches-bin/", "文件的目录")
tf.app.flags.DEFINE_string("cifar_tfrecords", "./data/cifar_tfrecords", "存进tfrecords的文件")


class CifarRead(object):
    """
    完成读取二进制文件, 写进tfrecords, 读取tfrecords
    :param file_list:
    :return:
    """

    def __init__(self, file_list):
        self.file_list = file_list

        # 定义图片的一些属性
        self.height = 32
        self.width = 32
        self.channel = 3
        self.label_bytes = 1
        self.image_bytes = self.height * self.width * self.channel
        self.bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self):
        # 1.构造文件队列
        file_queue = tf.train.string_input_producer(self.file_list)

        # 2.构造二进制文件阅读器
        reader = tf.FixedLengthRecordReader(self.bytes)
        key, value = reader.read(file_queue)
        print(value)

        # 3.解码: 二进制文件内容的解码
        label_image = tf.decode_raw(value, tf.uint8)
        print(label_image)

        # 4.分割出图片和标签数据, 特征值和目标值
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), tf.int32)
        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])
        # print(label, image)
        print("*" * 100)

        # 5.对图片的特征数据进行形状的改变[3072]--->[32,32,3]
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        image_reshape = tf.image.resize_images(image_reshape, (227, 227), method=0)
        print(label, image_reshape)
        print("*" * 100)

        # 5.读取多个数据,需要批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)
        print(image_batch, label_batch)

        return image_batch, label_batch

    def write_to_tfrecords(self, image_batch, label_batch):
        """
        将图片的特征值和目标值存进tfrecords
        :param image_batch: 10张图片的特征值
        :param label_batch: 10张图片的目标值
        :return: None
        """
        # 构造一个tfrecords存储器
        writer = tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)

        # 循环将所有样本写入文件,每张图片样本都要构造example协议
        for i in range(10):
            # 取出第i张图片的特征值和目标值
            # 要把张量转换成字符串, eval()
            image = image_batch[i].eval().tostring()
            label = label_batch[i].eval()[0]

            # 构造一个样本的example
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))

            # 写入单独的样本
            writer.write(example.SerializeToString())

        # 关闭
        writer.close()

        return None

    def read_from_tfrecords(self):
        # 1.构造文件队列
        file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecords])

        # 2.构造文件阅读器,读取内容example
        # value=一个样本的序列化example
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)

        # 3.解析example
        features = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
        # print(features["image"], features["label"])

        # 4.解码内容, 如果读取的内容是string,需要解码.如果是int64,float32不需要解码
        image = tf.decode_raw(features["image"], tf.uint8)

        # 固定图片的形状,方便批处理
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        label = tf.cast(features["label"], tf.int32)
        print(image_reshape, label)

        # 5.进行批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch


if __name__ == '__main__':
    file_name = os.listdir(FLAGS.cifar_dir)
    file_list = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]
    print(file_list)

    cf = CifarRead(file_list)
    image_batch, label_batch = cf.read_and_decode()

    # print("-" * 100)
    # image_batch, label_batch = cf.read_from_tfrecords()

    # 开启会话运行结果
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读文件的线程(开启子线程)
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 存进tfrecords文件
        print("开始存储")
        for i in range(60):
            print("第%d次开始" % i)
            cf.write_to_tfrecords(image_batch, label_batch)
        print("结束存储")

        # 打印读取的内容
        print(image_batch.shape)
        print(image_batch.eval())
        # print(sess.run([image_batch, label_batch]))

        # 更严谨的格式
        # try:
        #     while not coord.should_stop():
        #         # Run training steps or whatever
        #         print(sess.run([image_batch, label_batch]))
        #
        # except tf.errors.OutOfRangeError:
        #     print('Done training -- epoch limit reached')
        # finally:
        #     # When done, ask the threads to stop.
        #     coord.request_stop()

        # 回收
        coord.request_stop()
        coord.join(threads)
