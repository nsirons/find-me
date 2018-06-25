    # Ref: https://github.com/jagracar/OpenCV-python-tests/blob/master/OpenCV-tutorials/featureDetection/fast.py
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt

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


def draw_point(img, points, thick = 5, color = (0,0, 255)):
    points = np.array(points, dtype=int)
    img_copy = img #np.copy(img)
    for i in range(len(points)):
        cv2.circle(img_copy, (points[i][0], points[i][1]), thick, color, -1)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    return img_copy


def fast_corner(myimg_dir):
    img = cv2.imread(myimg_dir)
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create(threshold=25)
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img,None)
    keypts = np.array([])
    for point in kp:
        keypts = np.append(keypts, [*point.pt],axis=0)
    keypts = keypts.reshape((len(kp),2))

    # whitened = whiten(keypts)
    points =kmeans(keypts, 4)
    # print(points)
    #Implement nikitas dis(order)
    # points = points* np.std(keypts, axis=0)
    # img3 =draw_point(img, points[0], thick = 5, color = (0,0, 255))# cv2.drawKeypoints(img, kp, None, color=(255,0,0))
    # plt.imshow(img3)
    # plt.show()
    return keypts
if __name__ == "__main__":
    myimg_dir = 'C:\\Users\Daniel\Downloads\AE3200 Design Synthesis (201718 Q4)\Final\\find-me\cnn_model\\targets\\*.png'
    # myimg_dir2 ='C:\\Users\Daniel\Downloads\AE3200 Design Synthesis (201718 Q4)\Final\\find-me\data\cad_renders3\*.jpg'
    img_list = glob.glob(myimg_dir)#('cad_renders_dist\*.jpg')
    for img_dir in img_list:
        fast_corner(img_dir)

