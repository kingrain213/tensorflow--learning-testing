from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None
def main(_):
    #import data 下载数据
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    #create the model
    #输入样本
    x = tf.placeholder(tf.float32, [None, 784]) #高度不定，宽度28*28，一行为一幅图像，高度未知
    y_ = tf.placeholder(tf.float32, [None, 10])
    ##y = W * x + b 模型
    W = tf.Variable(tf.zeros([784, 10]), tf.float32)
    b = tf.Variable(tf.zeros([10]), tf.float32)
    y = tf.matmul(x, W) + b
    #define loss and optimizer
    #交叉熵验证 -y_logy
    cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)
    #模型保存加载工具
    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    total_accu = 9999

    for i in range(1000):
        #每100个图像一个batch，进行训练
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

        if i % 50 == 0: #可以改成，当当前的精度大于上一轮的精度了，才输出
            print (i)
            # one_hot后找出当前行最大值，判断是否相等
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            curr_accu = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            if total_accu < curr_accu: #k可以把前面的i%50去掉了，每次训练都跟上次比较大小
                total_accu = curr_accu
                curr_w, curr_b, cur_entroy = sess.run([W, b, cross_entroy], {x: batch_xs, y_: batch_ys})
                saver.save(sess, 'tmp/mnist_model.ckpt')
                print("***************changed accu %f loss %f" % (total_accu, cur_entroy))

    #测试模型精度

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
    #print (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
