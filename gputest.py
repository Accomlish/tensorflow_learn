import tensorflow as tf
import numpy as np

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    a = tf.placeholder(tf.int32)
    b = tf.placeholder(tf.int32)
    add = tf.add(a, b)
    sum666 = sess.run(add, feed_dict={a: 3, b: 4})
    print(sum666)