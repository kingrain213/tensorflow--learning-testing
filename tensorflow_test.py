import numpy as np
import tensorflow as tf
import time

#定义变量：
# 1）初始化时必须定义type、shape，但后期可以通过assing修改shape；
# 2）必须通过run()初始化，值才有意义
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

#占位符
#使用时必须feed_dict进行赋值，否则失败
x = tf.placeholder(tf.float32) #没有初始化shape，可以是任意类型
linear_model = W * x + b
y = tf.placeholder(tf.float32)

#loss
#reduce_sum：降维求和，没有axis参数时，整个数组各元素求和
#square：平方
loss = tf.reduce_sum(tf.square(linear_model - y))

#梯度下降法 学习率
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1,2,3,4]
y_train = [0, -1, -2, -3]

#模型保存加载工具
saver = tf.train.Saver()

#初始化所有全局变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train, feed_dict={x:x_train, y:y_train})  #开始训练
    print (i)
    curr_w, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b:%s loss:%s" % (curr_w, curr_b, curr_loss))

curr_w, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
#保存模型到tmp/model.ckpt
save_path = saver.save(sess, 'tmp/model.ckpt') 

print("W: %s b:%s loss:%s savepath: %s" %(curr_w, curr_b, curr_loss, save_path))

sess.close()
#待优化的地方，可以直接在训练结束保存
#也可以在循环当中，比较loss大小，循环保存当前最小值，删除上一次小值
