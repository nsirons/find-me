import os
from random import shuffle
import numpy as np
import skimage as sk
import tensorflow as tf
from skimage import io
from skimage.transform import resize

img_w, img_h = 240, 360
epochs = 10
batch_size = 10
base_path = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(base_path, 'inputs')
output_dir = os.path.join(base_path, 'targets')
log_dir = os.path.join(base_path, 'outputs_tf')


def image_generator(directory, extension, _batch_size, bw=False):
    return_list = []
    files = os.listdir(directory)
    shuffle(files)
    for _i, _file in enumerate(files):
        if extension in _file:
            file_path = os.path.join(directory, _file)
            if bw:
                new_image = sk.img_as_float(resize(io.imread(file_path, as_grey=True), (img_h, img_w)))
                return_list.append(np.reshape(new_image, (*new_image.shape, 1)))
            else:
                return_list.append(sk.img_as_float(resize(io.imread(file_path), (img_h, img_w))))
            if len(return_list) == batch_size:  # load n_batch images each time
                yield return_list
                return_list = []


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
                                         strides=[2, 2])

with tf.name_scope("loss"):
    error = deconv3 - target_layer
    # l2 is a regularizarion method err**2/2 to reduce loss error
    # -1 to get rid of one dimension
    # tf.product is the dot of the matrix dimensions to calculate all elements present
    print(tf.shape(error))
    loss = tf.reduce_mean(tf.nn.l2_loss(tf.reshape(error, [-1])))
    tf.summary.image("output diff.", error, 1)

with tf.name_scope("train"):
    train_operator = tf.train.AdamOptimizer(0.001).minimize(loss)

tf.summary.scalar("Loss", loss)
tf.summary.image("input images", input_layer, 1)
tf.summary.image("predicted images", deconv3, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # compile model and initialize all the placeholders
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir)
    writer.add_graph(sess.graph)

    global_it = 0
    for i in range(1, epochs + 1):
        input_gen = image_generator(input_dir, '.png', batch_size)
        target_gen = image_generator(output_dir, '.png', batch_size, bw=True)
        for j, (input_images, target_images) in enumerate(zip(input_gen, target_gen)):
            print("Iteration {} of epoch {} (total its = {})".format(j, i, global_it))
            if global_it % 1 == 0:
                s = sess.run(merged_summary, feed_dict={input_layer: input_images,
                                                        target_layer: target_images})
                writer.add_summary(s, global_it)
                writer.flush()
            train_operator.run(feed_dict={input_layer: input_images,
                                          target_layer: target_images})
            global_it += 1
            save_path = saver.save(sess, "./models/compressor_case_16.ckpt")
