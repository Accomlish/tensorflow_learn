"""
    Fetch and Feed

    1.Featch 在会话里可以执行多个操作

    2.Feed
"""
import tensorflow as tf
input1 = tf.constant(1.0)
input2 = tf.constant(2.0)
input3 = tf.constant(6.0)

#------------Featch--------
add = tf.add(input2,input3)

mul = tf.multiply(input1,add)

with tf.Session() as sess:
    #同时运行多个op，mul和add同时运行
    result = sess.run([mul,add])
    print(result)

#-------------Feed-------------
#创建占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
with tf.Session() as sess:
    #feed的数据以字典的形式传入
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))







