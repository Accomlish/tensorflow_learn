"""
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
print("number of train data is %d " % (mnist.train.num_examples))
print("number of test data is %d" %(mnist.test.num_examples))
#每个批次大小
batch_size = 100
#计算一共有多少批次
n_batch = mnist.train.num_examples//batch_size

#定义两个placheholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个神经网络Hidden Layer
W1 = tf.Variable(tf.random_normal([784,1000]))
b1 = tf.Variable(tf.zeros([1000]))
result1 = tf.matmul(x,W1)+b1

#创建输出层
W2 = tf.Variable(tf.random_normal([1000,10]))
b2 = tf.Variable(tf.zeros([10]))

prediction = tf.nn.softmax(tf.nn.relu(tf.matmul(result1,W2)+b2))

#交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

#结果存在放在一个bool列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter "+str(epoch)+"     Accuracy" + str(acc))
























