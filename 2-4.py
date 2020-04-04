"""
    tensorflow 简单使用案例
"""
import tensorflow as tf
import numpy as np
#使用numpy生成100个随机的点
x_data = np.random.rand(100)
y_data = x_data*1+3

#构造一个线性模型
b = tf.Variable(0.)#定义一个变量，初始值设为0.0
k = tf.Variable(0.)#定义一个变量，初始值设为0.0
#定义一个线性模型
y = k*x_data + b

#二次代价函数，计算误差
loss = tf.reduce_mean(tf.square(y_data-y))

#定义一个梯度下降法进行训练优化
optimizer = tf.train.GradientDescentOptimizer(0.2)

#最小化代价函数，使用上面定义的梯度下降法优化loss，减少误差
train = optimizer.minimize(loss)
#初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step,sess.run([k,b,loss]))


