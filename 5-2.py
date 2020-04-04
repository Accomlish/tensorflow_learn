# coding=utf-8
"""
    tensorbroad 学习
    (没怎么看懂，也没有敲代码)

    drop out 使用


    优化神经网络
    提高识别率
    搜写数字识别和softmax
    分类例子
    数据集：MNIST数据集，60000训练数据和10000测试数据组成
"""

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#载入数据
mnist = input_data.read_data_sets("MNIST_data",one_hot=True,source_url=
'http://yann.lecun.com/exdb/mnist/')
tf.logging.set_verbosity(old_v)
print("type of 'mnnist' is %s"% (type(mnist)))
print("number of train data is %d " % mnist.train.num_examples)
print("number of test data is %d" % mnist.test.num_examples)

#命名空间
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784],name='x-input')
    y = tf.placeholder(tf.float32, [None, 10],name='y-input')


#每个批次大小
batch_size = 100
#计算一共有多少批次
n_batch = mnist.train.num_examples



#定义一个placeholder设置drop out参数
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('layer'):
    with tf.name_scope('hidden_layer'):
        #创建一个神经网络Hidden Layer
        W1 = tf.Variable(tf.truncated_normal([784,1000],stddev=0.1))#正态分布初始值，标准差0.1
        b1 = tf.Variable(tf.zeros([1000])+0.1)
        #添加激活函数
        L1 = tf.tanh(tf.matmul(x,W1)+b1)
        #使用drop out
        L1_drop = tf.nn.dropout(L1,keep_prob)
    with tf.name_scope('out_layer'):
        #创建输出层
        W2 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
        b2 = tf.Variable(tf.zeros([10])+0.1)
    with tf.name_scope('prediction'):
        prediction = tf.nn.softmax(tf.matmul(L1,W2)+b2)

with tf.name_scope('loss'):
    #二次代价函数
    loss = tf.reduce_mean(tf.square(y - prediction))
with tf.name_scope('train'):
    #使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

#结果存在放在一个bool列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(1):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})#drop_out 设置1.0为全部神经元都用
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter "+str(epoch)+"     Accuracy" + str(acc))











































