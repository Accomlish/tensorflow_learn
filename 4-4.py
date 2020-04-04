"""
    Optimizer 优化器的使用
"""


import tensorflow as tf
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#载入数据
mnist = input_data.read_data_sets("MNIST_data",one_hot=True,source_url=
'http://yann.lecun.com/exdb/mnist/')
#tf.logging.set_verbosity(old_v)
print("type of 'mnnist' is %s"% (type(mnist)))
print("number of train data is %d " % mnist.train.num_examples)
print("number of test data is %d" % mnist.test.num_examples)
#每个批次大小
batch_size = 100
#计算一共有多少批次
n_batch = mnist.train.num_examples

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))

#使用梯度下降法
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#使用Adadelt优化器进行优化
train_step = tf.train.AdadeltaOptimizer(0.01).minimize(loss)




init = tf.global_variables_initializer()

#结果存在放在一个bool列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

gpu_options = tf.GPUOptions(allow_growth=True)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter "+str(epoch)+"     Accuracy" + str(acc))



















