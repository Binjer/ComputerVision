# 计算机视觉相关

## 常用的数据集
#### １.MNIST：主要用于图像分类。此数据集是以二进制存储的，有60000个训练样本集和10000个测试样本集，每个样本图像的宽高为28*28。下载地址：http://yann.lecun.com/exdb/mnist/index.html ，~12MB
#### ２.CIFAR-10：主要用于图像分类。包含10个类别，50,000个训练图像，10,000个测试图像，样本图像大小为32x32x3(3通道彩色图像)。下载地址：http://www.cs.toronto.edu/~kriz/cifar.html ，~170MB
#### ３.ImageNet：是目前深度学习图像领域应用得非常多的一个数据集，关于图像分类、定位、检测等研究工作大多基于此数据集展开。详细介绍和下载：http://www.image-net.org/about-stats ，~1TB
#### ４.PASCAL VOC：PASCAL VOC挑战赛是视觉对象的分类识别和检测的一个基准测试，提供了检测算法和学习性能的标准图像注释数据集和标准的评估系统。图片集包括20个目录：人类；动物（鸟、猫、牛、狗、马、羊）；交通工具（飞机、自行车、船、公共汽车、小轿车、摩托车、火车）；室内（瓶子、椅子、餐桌、盆栽植物、沙发、电视）。数据集图像质量好，标注完备，非常适合用来测试算法性能。下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html ，~2GB
#### 5.Open Image：是Google提供的一个包含~900万张图像URL的数据集，里面的图片通过标签注释被分为6000多类，该数据集中的标签要比ImageNet（1000类）包含更真实生活的实体存在。下载地址：https://github.com/openimages/dataset ,~1.5GB
#### 6.Youtube-8M：是谷歌开源的视频数据集，视频来自youtube，共计8百万个视频，总时长50万小时，4800类。为了保证标签视频数据库的稳定性和质量，谷歌只采用浏览量超过1000的公共视频资源。为了让受计算机资源所限的研究者和学生也可以用上这一数据库，谷歌对视频进行了预处理，并提取了帧级别的特征，提取的特征被压缩到可以放到一个硬盘中。下载地址：https://research.google.com/youtube8m/ ，~1.5TB

### 人脸识别相关
#### 1.FDDB：包含了数据集合和评测标准（benchmark），包含了2845张图像（5171人脸）。下载地址：http://vis-www.cs.umass.edu/fddb/
#### 2.WIDER FACE：是目前业界公开的数据规模最大(约40万人脸标注)、检测难度最高的人脸检测数据集之一。数据集中人脸尺寸大小变化、拍照角度引起的人脸姿态变化、人脸遮挡、化妆、光照等多种因素，给人脸检测带来了极大的挑战。下载地址：http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
#### 3.LFW:无约束自然场景人脸识别数据集，该数据集由13000多张全世界知名人士互联网自然场景不同朝向、表情和光照环境人脸图片组成，共有5000多人，其中有1680人有2张或2张以上人脸图片。每张人脸图片都有其唯一的姓名ID和序号加以区分。下载地址：http://vis-www.cs.umass.edu/lfw/index.html#download
#### 4.CelebA：包含10177个名人身份的202599张图片，并且都做好了特征标记（5个特征点）。下载地址：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
#### 5.AFLW：25993幅图像,每个人标定21个关键点。下载地址：http://neerajkumar.org/databases/lfpw/

### 其它：深度学习数据集收集网站：http://deeplearning.net/datasets/

