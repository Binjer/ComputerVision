人脸识别流程：
1 构造人脸数据库，通过facenet事先将每张人脸的特征提取出来并存储到本地
2 利用openCV提取实时视频帧，输入MTCNN提取人脸框，之后将提取的人脸框输入到facenet中进行特征提取
3 思路一:将提取的结果与数据库中的所有人脸特征进行比较，找到小于一定阈值的所有人脸特征，在这个阈值内找到最小的距离所对应的标签，给图片;如果所有结果都大于一定的阈值，则判定为非数据库内的人。
思路二：训练一个SVM分类器，判定输入的人脸是不是属于数据库中的某个人。