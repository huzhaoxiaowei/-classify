import tensorflow as tf
from debugpy._vendored.pydevd._pydev_bundle.pydev_umd import runfile

with tf.device('/GPU:0'):
    a = tf.random.normal([5000, 5000])
    b = tf.random.normal([5000, 5000])
    c = tf.matmul(a, b)
    print("GPU 测试完成:", c.shape)
#gg
