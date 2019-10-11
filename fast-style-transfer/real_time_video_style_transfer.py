"""风格化图片"""

import os
import numpy as np
from tensorflow.python.platform import gfile
import utils
import tensorflow as tf
import transform
import re
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# NETWORK_PATH = './pretrained-networks/dora-marr-network/'
# NETWORK_PATH = './pretrained-networks/rain-princess-network/'
NETWORK_PATH = './pretrained-networks/starry-night-network/'


def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def main():
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:

            video_capture = cv2.VideoCapture(0)
            video_capture.set(3, 300)
            video_capture.set(4, 300)

            load_model(NETWORK_PATH)

            c = 0
            while True:
                t1 = cv2.getTickCount()  # 记录当前时间，以时钟周期为单位
                ret, frame = video_capture.read()
                timeF = 1000
                if ret:
                    if c % timeF == 0:
                        content_image = np.array(frame)  # (height, width, channela =)
                        # content_image = utils.load_image(image)
                        reshaped_content_height = (
                                content_image.shape[0] - content_image.shape[0] % 4)  # 能被4整除，因为降采样将图片降到了原来的1/4
                        reshaped_content_width = (content_image.shape[1] - content_image.shape[1] % 4)
                        reshaped_content_image = content_image[:reshaped_content_height, :reshaped_content_width, :]
                        reshaped_content_image = np.ndarray.reshape(reshaped_content_image,
                                                                    (1,) + reshaped_content_image.shape)  # (1, H, W, C)
                        print("111111111")
                        # 创建placeholder并前向传播得到输出
                        img_placeholder = tf.placeholder(tf.float32, shape=reshaped_content_image.shape,
                                                         name='img_placeholder')
                        network = transform.net(img_placeholder)

                        sess.run(tf.global_variables_initializer())
                        prediction = sess.run(network, feed_dict={img_placeholder: reshaped_content_image})

                        print("22222222")
                        utils.save_image(prediction[0], "./results/my_video" + str(c) + ".jpg")
                        cv2.imshow("Input", frame)
                        cv2.imshow("Output", prediction[0])
                        print("33333333")
                    c += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            video_capture.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
