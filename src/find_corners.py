import cv2
import numpy as np
from src.GeneralImage import GeneralImage
import matplotlib.pyplot as plt
import os

import csv

# http://www.songho.ca/opengl/gl_projectionmatrix.html - Clear explanation of Perspective Frustum
# Gate information
CAMERA_RES = (1280, 720)
GATE_WIDTH = 1.4
FOV = 105 /2 * np.pi / 180
CAMERA_CENTER_PX = np.array([[1280 // 2 ], [ 720 // 2 ], [0]])
# SHIFT_Y = 0.15523613963039012
SHIFT_Y = 0.15
SHIFT_X = 0

# TODO : Change parameters (n,f to better values?)
# Camera planes parameters
n = 24.3 ** 10**-3  # Near plane
t = np.tan(FOV / 2 ) * n  # right
r = 1280 / 720 * t  # top
f = 10  # Far plane
scale = 1 / np.tan(FOV / 2)
# Perspective transformation matrix
M = np.array([[n / r, 0, 0, 0],
              [0, n / t, 0, 0],
              [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
              [0, 0, -1, 0]])
# M = np.array([[scale, 0, 0, 0],
#               [0, scale, 0, 0],
#               [0, 0, -f / (f - n), -f * n / (f - n)],
#               [0, 0, -1, 0]])

# Gate rotation matrix
rot_x = lambda alpha: np.array([[1, 0, 0],
                                [0, np.cos(alpha), -np.sin(alpha)],
                                [0, np.sin(alpha), np.cos(alpha)]])

rot_y = lambda beta: np.array([[np.cos(beta), 0, np.sin(beta)],
                               [0, 1, 0],
                               [-np.sin(beta), 0, np.cos(beta)]])

rot_z = lambda gamma: np.array([[np.cos(gamma),-np.sin(gamma), 0],
                                [np.sin(gamma), np.cos(gamma), 0],
                                [0, 0, 1]])


def get_corner(R, alpha, beta, POINT_REAL):
    c = alpha
    alpha = beta
    beta = c
    camera_pos = np.array([[0],
                           [0],
                           [-R]])
    point_real_rot = camera_pos + rot_x(beta*np.pi / 180) @ rot_y(alpha*np.pi / 180) @ (POINT_REAL - camera_pos)

    # point_real_rot = POINT_REAL - camera_pos
    # point_real_rot = POINT_REAL
    # print(rot_x(alpha*np.pi / 180) @ rot_y(beta*np.pi / 180) @ POINT_REAL)
    point_real_rot = np.vstack((point_real_rot * np.tan(FOV/2)*R * 2 / 1280, 1))
    # print(point_real_rot)
    c_c = M @ point_real_rot
    # print(c_c)
    corner = c_c[:3] / c_c[3]
    return corner


def generate_label_file(path, alphas, betas, Rs):
    # gi = GeneralImage(path)
    print("Generating corner position for {}".format(path))
    stored_labels = []
    counter = 0
    my_c = 0
    for R in Rs:
        # Real gate coordinates definition. (0,0,0) - camera

        for alpha in alphas:
            j = 0
            for beta in betas:
                SHIFT_X = 0.*np.sin(beta*2)
                POINT_REAL1 = np.array([[GATE_WIDTH / 2 - SHIFT_X], [GATE_WIDTH / 2 - SHIFT_Y], [-R]])
                POINT_REAL2 = np.array([[-GATE_WIDTH / 2 - SHIFT_X], [GATE_WIDTH / 2 - SHIFT_Y], [-R]])
                POINT_REAL3 = np.array([[-GATE_WIDTH / 2 - SHIFT_X], [-GATE_WIDTH / 2 - SHIFT_Y], [-R]])
                POINT_REAL4 = np.array([[GATE_WIDTH / 2 - SHIFT_X], [-GATE_WIDTH / 2 - SHIFT_Y], [-R]])

                x1, y1, z1 = get_corner(R,  alpha, beta, POINT_REAL1)
                x2, y2, z2 = get_corner(R,  alpha, beta, POINT_REAL2)
                x3, y3, z3 = get_corner(R,  alpha, beta, POINT_REAL3)
                x4, y4, z4 = get_corner(R,  alpha, beta, POINT_REAL4)
                # f = -R
                # x1 = f * POINT_REAL1[0]/POINT_REAL1[2]
                # x2 = f * POINT_REAL2[0] / POINT_REAL2[2]
                # x3 = f * POINT_REAL3[0] / POINT_REAL3[2]
                # x4 = f * POINT_REAL4[0] / POINT_REAL4[2]
                # y1 = f * POINT_REAL1[1] / POINT_REAL1[2]
                # y2 = f * POINT_REAL2[1] / POINT_REAL2[2]
                # y3 = f * POINT_REAL3[1] / POINT_REAL3[2]
                # y4 = f * POINT_REAL4[1] / POINT_REAL4[2]
                xs = np.array([x1, x4, x3, x2, x1])
                ys = np.array([y1, y4, y3, y2, y1])
                # print(xs)
                xs = CAMERA_CENTER_PX[0] + xs*CAMERA_RES[0] / 2 # TODO: change name
                ys = CAMERA_CENTER_PX[1] + ys*CAMERA_RES[1] / 2
                # print(x1-x2)
                stored_labels.append({"Name": "img_{:04d}".format(counter), "R":R, "alpha": alpha, "beta":beta, "gate":1})   # TODO: change gate status for empty pictures
                for i in range(4):
                    stored_labels[-1]["x{}".format(i)] = xs[i][0]
                    stored_labels[-1]["y{}".format(i)] = ys[i][0]

                each = 5
                if my_c % each == 0:

                    plt.subplot(4, 3, my_c//each+1)
                    # print((j//10+1))
                    plt.plot(xs, ys, color='red')
                    # plt.subplot(6,2,j//6 + 2)
                    gi = GeneralImage(path + '/' + "GateRenders_00{:02d}.jpg".format(j+1))
                    # print("img{}_alpha{}_beta{}_R{}.png".format(my_c,alpha,beta,R))
                    # gi = GeneralImage(path + '/' + "img{}_alpha{}_beta{}_R{}.png".format(my_c,alpha,beta,R))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    img = gi.rgb()
                    cv2.putText(img, 'c0', (xs[0],ys[0]), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, 'c1', (xs[1],ys[1]), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, 'c2', (xs[2],ys[2]), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, 'c3', (xs[3],ys[3]), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                    plt.imshow(img)
                my_c += 1
                j += 1
                counter += 1
    # gi = GeneralImage(path+'/'+"GateRenders_0001.jpg")
    # plt.imshow(gi.rgb())
    plt.show()

    print("Storing data to: {}".format("{}/{}".format(path, "corners.txt")))
    with open("{}/{}".format(path, "corners.txt"), mode='w+') as csvfile:
        fieldnames = stored_labels[-1].keys()
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        csvwriter.writeheader()
        for label in stored_labels:
            csvwriter.writerow(label)

if __name__ == "__main__":
    path = "../data/cad_renders3"
    # path1 =  "data/ale_inputs"
    # alpha1= [0,45,90]
    # beta1 = [0,45,90]
    alpha = [0]
    beta = np.linspace(0, 90, 60)
    R = [3]
    # generate_label_file(path, alpha, beta, R)
    generate_label_file(path, alpha, beta, R)
    # find_axis(path)
