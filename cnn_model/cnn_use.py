import tensorflow as tf
import time
import cv2
from scipy.cluster.vq import kmeans
import os
import numpy as np
import matplotlib.pyplot as plt
import random


class CNN_gate:

    def __init__(self):
        self.sess = None
        self.deconv3 = None
        self.h = None
        self.w = None

    def restore_model(self, path):
        self.h, self.w = 288, 512

        self.input_layer = tf.placeholder(tf.float32, shape=[None, self.h, self.w, 3],
                                     name='input')  # First node where input is generated 3 is for rgb
        self.target_layer = tf.placeholder(tf.float32, shape=[None, self.h, self.w, 1],
                                      name='targets')  # Tengo  que meterle las imagenes del target desde fuera 1 is for b/w color


        with tf.name_scope("convolution"):  # tf.name_scope is used for graphic visualization
            conv1 = tf.layers.conv2d(inputs=self.input_layer,
                                     filters=32,
                                     kernel_size=(3, 3),
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
                                                 )
            self.deconv3 = deconv3
            saver = tf.train.Saver()
            self.sess = tf.Session()
            saver.restore(self.sess, path)

    def find_gate(self, img_path):
        img = cv2.resize(cv2.imread(img_path), (self.w, self.h)).reshape((1, self.h, self.w, 3))
        target = np.zeros((1, self.h, self.w, 1))
        cnn_image = self.sess.run(self.deconv3, feed_dict={self.input_layer:img.astype(float),
                                                            self.target_layer:target.astype(float)})
        # with self.sess as sess:
        # cnn_image = self.deconv3.eval(feed_dict={self.input_layer:img.astype(float),
        #                                     self.target_layer:target.astype(float)})

        # plt.subplot(11)
        # plt.imshow(cnn_image.reshape((self.h, self.w)))
        # plt.show()
        # cnn_image = np.clip(cnn_image.reshape((self.h, self.w)),1,255).astype(int).astype(np.uint8)

        # print(cnn_image.shape)
        # print(type(cnn_image))
        # assert False
        blur = cv2.GaussianBlur(cnn_image.reshape((self.h, self.w)), (5, 5), 0)
        # plt.hist(blur.flatten(), bins=50)
        # plt.show()
        _, cnn_image_th = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
        cnn_image_th = cv2.GaussianBlur(cnn_image_th, (5, 5), 0)
        # plt.subplot(132)
        # plt.imshow(cnn_image_th, 'gray')

        # ret3, cnn_image_th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # plt.subplot(133)
        # plt.imshow(cnn_image_th, 'gray')
        # plt.show()
        # # TODO: detect if there is gate
        # # print(cnn_image_th)
        corners, shit = self.fast_corner(cnn_image_th.astype(np.uint8))  # TODO: order corners
        fast_all = self.draw_point(cnn_image_th, shit)
        cnn_image_th = self.draw_point(cnn_image_th, corners)

        return cnn_image_th, corners, cnn_image.reshape((self.h, self.w)), fast_all

    def fast_corner(self, img):
        fast = cv2.FastFeatureDetector_create(threshold=16)
        fast.setNonmaxSuppression(0)
        kp = fast.detect(img, None)
        keypts = cv2.KeyPoint.convert(kp)
        try:
            points = kmeans(keypts, 4)
            # print(points[0], points[0].shape)
        except ValueError:
            points = np.zeros((1,4,2), dtype=np.uint8)
            # print("HELLO0,", points[0].shape)
        return points[0], keypts

    @staticmethod
    def draw_point(img, points, thick=5, color=(0, 0, 255)):
        points = np.array(points, dtype=int)
        img_copy = img  # np.copy(img)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
        for i in range(len(points)):
            cv2.circle(img_copy, (points[i][0], points[i][1]), thick, color, -1)

        return img_copy

if __name__ == "__main__":
    path = "/home/kit/projects/find-me/cnn_model/models_final/compressor_case_16.ckpt"
    main = "/home/kit/projects/find-me/data/cnn_data4/inputs_mod"
    t = CNN_gate()
    t.restore_model(path)
    images = []
    corners = []
    lst = os.listdir(main)
    random.shuffle(lst)
    for i,im in enumerate(lst[:100]):
        # img_path = "/home/kit/projects/find-me/data/cnn_data3/inputs_new/GateRendersApproach_0020random01609_compression010.jpg"
        img_path = os.path.join(main, im)
        # target_or = cv2.imread("../data/cnn_data3/targets_new/GateRendersApproach_0020random01609_compression010.jp

        img, corner,or_img,f = t.find_gate(img_path)
        images.append((or_img, img, f))
        corners.append(corner)
        print(i,im)
    i = 0
    while True:
        # k = cv2.waitKey(3000) & 0xff
        # cv2.imshow("Template matching animation", images[i % len(images)])
        plt.subplot(131)
        plt.imshow(images[i % len(images)][0])
        plt.subplot(132)
        plt.imshow(images[i % len(images)][1])
        plt.subplot(133)
        plt.imshow(images[i % len(images)][2])
        plt.show()

        # if k == 27:  # wait for ESC key to exit
        #     break
        i += 1
    cv2.destroyAllWindows()
