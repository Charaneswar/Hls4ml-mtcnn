import tensorflow as tf
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model-20180408-102900.meta')
    saver.restore(sess, "model-20180408-102900.ckpt")