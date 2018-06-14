import tensorflow as tf
# from conv_deconv_model import deconv3
import cv2

tf.reset_default_graph()
# with tf.device("/device:CPU:0"):
img_h, img_w = 288, 512
input_layer = tf.placeholder(tf.float32, shape=[None, img_h, img_w, 3], name='input')  # First node where input is generated 3 is for rgb
target_layer = tf.placeholder(tf.float32, shape=[None, img_h, img_w, 1], name='targets')  # Tengo  que meterle las imagenes del target desde fuera 1 is for b/w color

with tf.name_scope("convolution"):  # tf.name_scope is used for graphic visualization
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=(5, 5),
                             padding='SAME',
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=(2, 2),
                                    strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=32,
                             kernel_size=(5, 5),
                             padding='SAME',
                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=(2, 2),
                                    strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2,
                             filters=32,
                             kernel_size=(5, 5),
                             padding='SAME',
                             activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=(2, 2),
                                    strides=2)

with tf.name_scope("deconvolution"):
    deconv1 = tf.layers.conv2d_transpose(inputs=pool3,
                                         filters=32,
                                         kernel_size=(5, 5),
                                         padding='SAME',
                                         strides=[2, 2])

    deconv2 = tf.layers.conv2d_transpose(inputs=deconv1,
                                         filters=16,
                                         kernel_size=(5, 5),
                                         padding='SAME',
                                         strides=[2, 2])

    deconv3 = tf.layers.conv2d_transpose(inputs=deconv2,
                                         filters=1,
                                         kernel_size=(5, 5),
                                         padding='SAME',
                                         strides=[2, 2],
                                         name="dany_yag")
# Create some variables.
# v1 = tf.get_variable("v1", shape=[3])
# v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
# img = cv2.imread("C:\\Users\\kit\\projects\\find-me\\data\\cnn_inputs\\img788_alpha0.77958465_beta0.01985693_R93.68667.jpg")
img = cv2.imread("C:\\Users\kit\projects\\find-me\data\cad_renders2\GateRenders_0001.jpg")
img = cv2.resize(img, (img_w, img_h))
# target = cv2.imread("C:\\Users\\kit\\projects\\find-me\\data\\cnn_targets\\img788_alpha0.77958465_beta0.01985693_R93.68667.jpg",0)
import numpy as np
target = np.zeros((img_h, img_w))
img = img.reshape((1,*img.shape))
target = target.reshape((1,*target.shape,1))
import matplotlib.pyplot as plt
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "models/compressor_case_16.ckpt")
  test = deconv3.eval(feed_dict={input_layer:img.astype(float),
                                target_layer:target.astype(float)})
  plt.imshow(test.reshape((img_h,img_w)))
  plt.show()
  print("Model restored.")
  # Check the values of the variables
  # print("v1 : %s" % v1.eval())
  # print("v2 : %s" % v2.eval())