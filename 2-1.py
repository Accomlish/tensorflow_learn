"""
    创建一个Graph，然后启动
"""
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])

product = tf.matmul(m1,m2)
print(product)

#定义一个会话，启动默认Graph
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

#定于会话方法2：
with tf.Session() as sess:
    result = sess.run(product)
    print(result)