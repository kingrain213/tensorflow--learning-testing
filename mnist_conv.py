from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

'''''
权重初始化
初始化为一个接近0的很小的正数
'''

def weight_variable(shape):
    #正态分布 标准差为0.1，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    #创建一个结构为shape的矩阵，行列、初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

'''''
卷积和池化，使用卷积步长为1（stride size）,0边距（padding size）
池化用简单传统的2x2大小的模板做max pooling
'''
#图像卷积 图像*滤波器权重
def conv2d(x, W):
    #卷积便利个方向步数为1，SAME：边缘外自动补0，遍历相乘
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#池化采样层 输入原始数据，输出shape变小
def max_pool_2x2(x):
    #池化卷积结果（conv2d）池化层采用kernel大小为2*2，步长也为2，周围补0，取最大值，数据量缩小了4倍
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def train():
    #计算开始时间
    start = time.clock()
    #MNIST数据输入
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #声明一个占位符，None表示输入图像数量不定，784=28*28，图片分辨率
    x = tf.placeholder(tf.float32, [None, 784])  #图像输入向量
    W = tf.Variable(tf.zeros([784, 10]))  #权重，初始化值为全零
    b = tf.Variable(tf.zeros([10]))  #偏置，初始化值为全零


    # 第一层卷积，由一个卷积接一个maxpooling完成，卷积在每个
    # 5x5的patch中算出32个特征。
    # 卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，
    # 接着是输入的通道数目，最后是输出的通道数目。
    
    W_conv1 = weight_variable([5,5,1,32]) #参数1、2表示卷积核大小，参数3表示通道数，参数4表示卷积核数目，最后输出多少个卷积特征图像
    # 而对于每一个输出通道都有一个对应的偏置量。
    b_conv1 = bias_variable([32])

    #把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3,-1表示个数不定)。
    x_image = tf.reshape(x, [-1, 28, 28, 1]) #最后一维代表通道数目，如果是rgb则为3
    # x_image权重向量卷积，加上偏置项，之后应用ReLU函数，之后进行max_polling
    # 图片乘以卷积核，并加上偏执量，卷积结果28x28x32  
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # 池化结果14x14x32 卷积结果乘以池化卷积核
    h_pool1 = max_pool_2x2(h_conv1)

    #实现第二层卷积
    # 输入32通道图像，每个5x5的patch会得到64个特征，对应64个偏执
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #卷积结果14*14*64
    h_pool2 = max_pool_2x2(h_conv2) #池化结果7*7*64

    # 密集连接层 全连接
    '''''
    图片尺寸变为7x7*64，加入有1024个神经元的全连接层，把池化层输出张量reshape成向量
    乘上权重矩阵，加上偏置，然后进行ReLU
    '''
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout， 用来防止过拟合 #加在输出层之前，训练过程中开启dropout，测试过程中关闭
    # dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，
    # 升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
    # 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，
    # 程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0  
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层, 添加softmax层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    # 最后的分类，结果为1*1*10 softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类 
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    #训练和评估模型
    #ADAM优化器来做梯度最速下降,feed_dict 加入参数keep_prob控制dropout比例
    # 类别是0-9总共10个类别，对应输出分类结果
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # 定义交叉熵为loss函数 

    # 使用adam优化器来以0.0001的学习率来进行微调
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化

    # 判断预测标签和实际标签是否匹配
    correct_predicition = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))

    # 启动创建的模型，并初始化变量
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # 开始训练模型，循环训练20000次
    for i in range(2000):
        batch = mnist.train.next_batch(50) #batch 大小设置为50
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session = sess, feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0})
            print("step %d, train_accuracy %g" % (i, train_accuracy))

        #神经元输出保持不变的概率 keep_prob 为0.5
        train_step.run(session=sess, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

    # 神经元输出保持不变的概率 keep_prob 为 1，即不变，永远保持输出
    print("test accuracy %g" % accuracy.eval(session=sess,feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

    end = time.clock()
    print("running time is %g s") % (end - start)

if __name__ == '__main__':
    train()
