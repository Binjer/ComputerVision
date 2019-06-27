import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from scipy import misc
from MTCNN_detection.MtcnnDetector import MtcnnDetector
from MTCNN_detection.detector import Detector
from MTCNN_detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
from sklearn.svm import SVC


def face2database(picture_path, model_path, database_path, batch_size=64, image_size=160):
    # 提取特征到数据库  [']
    # picture_path为人脸文件夹的所在路径
    # model_path为facenet模型路径
    # database_path为人脸数据库路径
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dataset = facenet.get_dataset(picture_path)  # 数据库中所有类别的列表
            # [<facenet.ImageClass object>, <facenet.ImageClass object>, <facenet.ImageClass object>, <facenet.ImageClass object>, <facenet.ImageClass object>, <facenet.ImageClass object>]
            # print(dataset)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            # print("图片：", paths)
            # print("标签", labels)  # 两者一一对应
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model_path)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")  # 网络的输出，是后续SVM分类器的输入
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)  # 图片总数
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))  # 分成多少批处理
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]

                # 两个False对应的参数是do_random_crop, do_random_flip
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            np.savez(database_path, emb=emb_array, lab=labels)  # emb_array里存放的是图片特征，labels为对应的标签

            print("数据库特征提取完毕！")


def ClassifyTrainSVC(database_path, SVMpath):
    # database_path为人脸数据库
    # SVMpath为分类器储存的位置
    Database = np.load(database_path)
    name_lables = Database['lab']
    embeddings = Database['emb']
    name_unique = np.unique(name_lables)  # 去重并排序后输出
    # print(name_lables)

    # 对标签进行数字表示
    labels = []
    for i in range(len(name_lables)):
        for j in range(len(name_unique)):
            if name_lables[i] == name_unique[j]:  # 名字等于对应位置的类别名, 则标记为j类
                labels.append(j)
    # print(labels)

    print('Training classifier')
    model = SVC(kernel='linear', probability=True)  # SVM如何解决多分类问题：一对一；一对多；有向无环图
    model.fit(embeddings, labels)
    with open(SVMpath, 'wb') as outfile:
        pickle.dump((model, name_unique), outfile)
        print('Saved classifier model to file "%s"' % SVMpath)


# 图片预处理阶段
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def prewhiten(x):
    """
    图像白化处理
    减少由于环境照明强度、物体反射、拍摄相机等因素的影响，获得图像包含的不受外界影响的恒定信息
    """
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))  # x.size=height * width * channel
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def load_image(image_old, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    if image_old.ndim == 2:
        image_old = to_rgb(image_old)
    if do_prewhiten:
        image_old = prewhiten(image_old)
    image_old = crop(image_old, do_random_crop, image_size)
    image_old = flip(image_old, do_random_flip)
    return image_old


def RTrecognization(facenet_model_path, SVMpath, database_path):
    # facenet_model_path为facenet模型路径
    # SVCpath为SVM分类模型路径
    # database_path为人脸库数据
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(facenet_model_path)
            with open(SVMpath, 'rb') as infile:
                (classifymodel, class_names) = pickle.load(infile)
            print('Loaded classifier model from file "%s"' % SVMpath)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            Database = np.load(database_path)

            test_mode = "onet"
            thresh = [0.9, 0.6, 0.7]
            min_face_size = 24
            stride = 2
            slide_window = False
            shuffle = False
            # vis = True
            detectors = [None, None, None]
            prefix = ['./data/MTCNN_model/PNet_landmark/PNet', './data/MTCNN_model/RNet_landmark/RNet',
                      './data/MTCNN_model/ONet_landmark/ONet']
            epoch = [18, 14, 16]
            model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
            PNet = FcnDetector(P_Net, model_path[0])
            detectors[0] = PNet
            RNet = Detector(R_Net, 24, 1, model_path[1])
            detectors[1] = RNet
            ONet = Detector(O_Net, 48, 1, model_path[2])
            detectors[2] = ONet
            mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                           stride=stride, threshold=thresh, slide_window=slide_window)

            video_capture = cv2.VideoCapture(0)
            # video_capture.set(3, 340)
            # video_capture.set(4, 480)
            video_capture.set(3, 640)
            video_capture.set(4, 480)
            corpbbox = None
            while True:
                t1 = cv2.getTickCount()  # 记录当前时间，以时钟周期为单位
                ret, frame = video_capture.read()
                if ret:
                    image = np.array(frame)  # (height, width, channela =)
                    img_size = np.array(image.shape)[0:2]
                    boxes_c, landmarks = mtcnn_detector.detect(image)
                    # print("boxes_c.shape", boxes_c.shape)  # (1,5) 5个特征点
                    # print("boxes_c", boxes_c)
                    # print(img_size)

                    t2 = cv2.getTickCount()
                    t = (t2 - t1) / cv2.getTickFrequency()
                    fps = 1.0 / t
                    # print("MTCNN每秒处理%s帧" % fps)

                    for i in range(boxes_c.shape[0]):
                        bbox = boxes_c[i, :4]  # 检测出的人脸区域，左上x，左上y，右下x，右下y
                        score = boxes_c[i, 4]  # 检测出人脸区域的得分
                        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

                        # 从图片上裁剪比MTCNN返回的人脸框稍微大点的区域作为FaceNet的输入
                        x1 = np.maximum(int(bbox[0]) - 16, 0)
                        y1 = np.maximum(int(bbox[1]) - 16, 0)
                        x2 = np.minimum(int(bbox[2]) + 16, img_size[1])  # 宽
                        y2 = np.minimum(int(bbox[3]) + 16, img_size[0])  # 高
                        crop_img = image[y1:y2, x1:x2]  # 裁剪原图得到输入图片

                        scaled = misc.imresize(crop_img, (160, 160), interp='bilinear')
                        img = load_image(scaled, False, False, 160)  # 做了归一化，即每个通道的值都除以255
                        img = np.reshape(img, (-1, 160, 160, 3))

                        feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                        embvecor = sess.run(embeddings, feed_dict=feed_dict)
                        # 获得检测人脸的embedding向量
                        embvecor = np.array(embvecor)

                        # 利用人脸特征与数据库中所有人脸进行一一比较的方法
                        # tmp=np.sqrt(np.sum(np.square(embvecor-Database['emb'][0])))
                        # tmp_lable=Database['lab'][0]
                        # for j in range(len(Database['emb'])):
                        #     t=np.sqrt(np.sum(np.square(embvecor-Database['emb'][j])))
                        #     if t<tmp:
                        #         tmp=t
                        #         tmp_lable=Database['lab'][j]
                        # print(tmp)

                        # 利用SVM对人脸特征进行分类
                        predictions = classifymodel.predict_proba(embvecor)
                        best_class_indices = np.argmax(predictions, axis=1)
                        tmp_lable = class_names[best_class_indices]
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        # 如果小于某个阈值，预测为其它
                        if best_class_probabilities < 0.4:
                            tmp_lable = "others"
                        print("预测为%s的概率为%s" % (tmp_lable, best_class_probabilities))

                        # 在视频中绘制人脸框
                        cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                                      (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                        # 显示人名
                        cv2.putText(frame, '{0}'.format(tmp_lable), (corpbbox[0], corpbbox[1] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 2)

                    # cv2.putText(img, 左上角文字, 左上角坐标, 字体, 字体大小, 颜色, 字体粗细)
                    cv2.putText(frame, '{:.4f}'.format(t) + "Processing per second:" + '{:.3f}'.format(fps), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 255), 2)

                    # 绘制5个标记点
                    for i in range(landmarks.shape[0]):
                        for j in range(len(landmarks[i]) // 2):
                            # cv2.circle(img, 圆心坐标, 半径, 颜色(B,G,R))
                            cv2.circle(frame, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2,
                                       (0, 255, 0))
                            # time end

                    cv2.imshow("", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:

                    print('device not find')
                    break
            video_capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    picture_path = "/home/binjer/Desktop/MyNote/DeepLearning/MyProj_DL/Real-time-face-recognition/face_database"
    model_path = "/home/binjer/Desktop/MyNote/DeepLearning/MyProj_DL/Real-time-face-recognition/face_models/20180402-114759"
    database_path = "/home/binjer/Desktop/MyNote/DeepLearning/MyProj_DL/Real-time-face-recognition/MyDatabase.npz"
    SVMpath = "/home/binjer/Desktop/MyNote/DeepLearning/MyProj_DL/Real-time-face-recognition/face_models/SVMmodel.pkl"

    # face2database(picture_path, model_path, database_path)  # 第一步 将图像信息embedding
    # ClassifyTrainSVC(database_path, SVMpath)  # 第二步 将第一步的结果进行训练SVC

    RTrecognization(model_path, SVMpath, database_path)  # 第三步 实时检测
