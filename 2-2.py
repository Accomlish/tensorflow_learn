# coding=utf-8
"""
    tensorflow 变量的使用
    Variable 是变量
    constant 是常量
"""
import tensorflow as tf

# x = tf.Variable([1,2])
# a = tf.constant([3,3])
# #增加一个减法的op
# sub = tf.subtract(x,a)
# #增加一个加法的op
# add = tf.add(x,a)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(sub))
#


#创建一个变量，初始化为0，名字为counter
state = tf.Variable(0,name='counter')

#创建一个op，作用是使state+1
new_value = tf.add(state,1)
#赋值op
update = tf.assign(state,new_value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state),end=" ")





