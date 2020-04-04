import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

#载入数据
mnist = input_data.read_data_sets("MNIST_data",one_hot=True,source_url=
'http://yann.lecun.com/exdb/mnist/')

#输入图片是28*28
n_inputs = 28 #输入一行，一行有28个数据
max_time = 28#一共28行
lstm_size = 100#隐层单元
n_classes = 10 #10个分类
batch_size = 50 #每批次50个样本
n_batch = mnist.train.num_examples // batch_size #计算一共有多少批次

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
#
weight = tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev=0.1))
biase = tf.Variable(tf.constant(0.1,shape=[n_classes]))

#定义RNN
def RNN(X,weight,biase):
    inputs = tf.reshape(X,[-1,max_time,n_inputs])
    #定义LSTM基本CELL
    lstm_cell = rnn.BasicLSTMCell(lstm_size)
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1],weight)+biase)
    return results
#RNN的返回结果
prediction = RNN(x,weight,biase)
#损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
#使用AdmOptimizer优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个bool列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter  "+str(epoch) + "Testing Accuracy=" + str(acc))



