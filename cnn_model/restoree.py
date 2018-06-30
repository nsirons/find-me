import tensorflow as tf
# from conv_deconv_model import deconv3
import cv2
import time
# tf.reset_default_graph()
g = tf.Graph()
run_meta = tf.RunMetadata()
import keras.backend as K
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt

with g.as_default():
    # with tf.device("/device:CPU:0"):
    img_h, img_w = 288, 512
    input_layer = tf.placeholder(tf.float32, shape=[None, img_h, img_w, 3], name='input')  # First node where input is generated 3 is for rgb
    target_layer = tf.placeholder(tf.float32, shape=[None, img_h, img_w, 1], name='targets')  # Tengo  que meterle las imagenes del target desde fuera 1 is for b/w color

    with tf.name_scope("convolution"):  # tf.name_scope is used for graphic visualization
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=32,
                                 kernel_size=(3, 3),
                                 padding='SAME',
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=(2, 2),
                                        strides=2)

        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=32,
                                 kernel_size=(3, 3),
                                 padding='SAME',
                                 activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=(2, 2),
                                        strides=2)

        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=32,
                                 kernel_size=(3, 3),
                                 padding='SAME',
                                 activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                        pool_size=(2, 2),
                                        strides=2)

    with tf.name_scope("deconvolution"):
        deconv1 = tf.layers.conv2d_transpose(inputs=pool3,
                                             filters=32,
                                             kernel_size=(3, 3),
                                             padding='SAME',
                                             strides=[2, 2])

        deconv2 = tf.layers.conv2d_transpose(inputs=deconv1,
                                             filters=16,
                                             kernel_size=(3, 3),
                                             padding='SAME',
                                             strides=[2, 2])

        deconv3 = tf.layers.conv2d_transpose(inputs=deconv2,
                                             filters=1,
                                             kernel_size=(3, 3),
                                             padding='SAME',
                                             strides=[2, 2],
                                             )
    # Create some variables.
    # v1 = tf.get_variable("v1", shape=[3])
    # v2 = tf.get_variable("v2", shape=[5])

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    # img = cv2.imread("C:\\Users\\kit\\projects\\find-me\\data\\cnn_inputs\\img788_alpha0.77958465_beta0.01985693_R93.68667.jpg")
    # img = cv2.imread("C:\\Users\kit\projects\\find-me\data\cad_renders2\GateRenders_0001.jpg")
    img_or = cv2.imread("../data/cnn_data5/inputs_bg/GateRendersPassBy_0015random00243_blur010.jpg")
    target_or = cv2.imread("../data/cnn_data5/targets_bg/GateRendersPassBy_0015random00243_blur010.jpg", 0)

    img = cv2.resize(img_or, (img_w, img_h))
    # target = cv2.imread("C:\\Users\\kit\\projects\\find-me\\data\\cnn_targets\\img788_alpha0.77958465_beta0.01985693_R93.68667.jpg",0)
    import numpy as np
    # target = np.zeros((img_h, img_w))
    img = img.reshape((1,*img.shape))
    target = target_or.reshape((1,*target_or.shape,1))
    import matplotlib.pyplot as plt


    with tf.Session() as sess:
      K.set_session(sess)
      # Restore variables from disk.
      saver.restore(sess, "models_3/compressor_case_16.ckpt")
      t1 = time.perf_counter()
      test = deconv3.eval(feed_dict={input_layer:img.astype(float),
                                    target_layer:target.astype(float)})
      # test = test.reshape((img_h, img_w))
      print("TIME: ",time.perf_counter() - t1)
      print(conv1)
      print(conv2)
      print(conv3)
      print(deconv1)
      print(deconv2)
      print(deconv3)
      # test = deconv3.eval(feed_dict={input_layer:img.astype(float),
      #                               target_layer:target.astype(float)})
      # test = deconv3.eval(feed_dict={input_layer:img.astype(float),
      #                               target_layer:target.astype(float)})

      # opts = tf.profiler.ProfileOptionBuilder.float_operation()
      # flops = tf.profiler.profile(g, run_meta=run_meta, cmd='graph', options=opts)
      # opts = tf.profiler.ProfileOptionBuilder.float_operation()
      # flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
      #
      # opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
      # params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
      # opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
      # params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

      # print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
      # if flops is not None:
      #     print('Flops should be ~', 2 * 25 * 16 * 9)
      #     print('25 x 25 x 9 would be', 2 * 25 * 25 * 9)  # ignores internal dim, repeats first
      #     print('TF stats gives', flops.total_float_ops)
      plt.subplot(221)
      plt.imshow(cv2.cvtColor(img_or, cv2.COLOR_RGB2BGR))
      plt.subplot(222)
      plt.imshow(target_or, 'gray')
      plt.subplot(223)
      plt.imshow(test.reshape((img_h, img_w)), 'gray')
      asd, test = cv2.threshold(test.reshape((img_h, img_w)), 180, 255, cv2.THRESH_BINARY)
      test = test.astype(np.uint8)
      # plt.show()
      # plt.subplot(224)
      # plt.imshow(test, 'gray')
      # cv2.imwrite('dany_yag.png', test)
      # plt.show()
      # print("Model restored.")
    # Ref: https://github.com/jagracar/OpenCV-python-tests/blob/master/OpenCV-tutorials/featureDetection/fast.py



# def neighbor(start_point, points, radius_thresh):
#     prev_radius = radius_thresh
#     next_neighbor = None
#     next_neighbors =[]
#     for other_point in points:
#         radius = np.linalg.norm(start_point - other_point)
#         if radius < prev_radius:
#             next_neighbor = other_point
#             prev_radius = radius
#             # next_neighbors.append(next_neighbor)
#
#     return next_neighbor


def draw_point(img, points, thick = 5, color = (0, 0, 255)):
    points = np.array(points, dtype=int)
    img_copy = img #np.copy(img)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    for i in range(len(points)):
        cv2.circle(img_copy, (points[i][0], points[i][1]), thick, color, -1)

    return img_copy


def fast_corner(img):
    # img = cv2.imread(myimg_dir)
    # Initiate FAST object with default values
    # t1 =
    fast = cv2.FastFeatureDetector_create(threshold=16)
    # print(img.astype(np.uint8))
    # plt.imshow(img.astype(np.uint8), "gray")
    # plt.show()
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)
    keypts = np.array([])
    t2 = time.perf_counter()
    for point in kp:
        keypts = np.append(keypts, [*point.pt], axis=0)
    print(cv2.KeyPoint.convert(kp))
    keypts = keypts.reshape((len(kp),2))
    print(keypts)
    # whitened = whiten(keypts)
    points =kmeans(keypts, 4)
    print(time.perf_counter() - t2)
    #Implement nikitas dis(order)
    # points = points* np.std(keypts, axis=0)
    img3 = draw_point(img, points[0], thick=5, color=(255,0, 0))#
    # img = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
    print(img3.shape)
    plt.subplot(224)
    plt.imshow(img3)
    plt.show()


# myimg_dir = 'C:\\Users\Daniel\Downloads\AE3200 Design Synthesis (201718 Q4)\Final\\find-me\cnn_model\\targets\\*.png'
# myimg_dir2 ='C:\\Users\Daniel\Downloads\AE3200 Design Synthesis (201718 Q4)\Final\\find-me\data\cad_renders3\*.jpg'
# img_list = glob.glob(myimg_dir2)#('cad_renders_dist\*.jpg')
# for img_dir in img_list:
fast_corner(test)

  # Check the values of the variables
  # print("v1 : %s" % v1.eval())
  # print("v2 : %s" % v2.eval())