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
FOV = 105 * np.pi / 180
CAMERA_CENTER_PX = np.array([[1280 // 2 ], [ 450 ], [0]])
SHIFT_Y = 0.15523613963039012
SHIFT_X = 0

# TODO : Change parameters (n,f to better values?)
# Camera planes parameters
n = 24.3 ** 10**-3  # Near plane
t = np.tan(FOV / 2) * n  # right
r = 1280 / 720 * t  # top
f = 10  # Far plane

# Perspective transformation matrix
M = np.array([[n / r, 0, 0, 0],
              [0, n / t, 0, 0],
              [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
              [0, 0, -1, 0]])

# Gate rotation matrix
rot_x = lambda alpha: np.array([[1, 0, 0],
                                [0, np.cos(alpha), -np.sin(alpha)],
                                [0, np.sin(alpha), np.cos(alpha)]])

rot_y = lambda beta: np.array([[np.cos(beta), 0, np.sin(beta)],
                               [0, 1, 0],
                               [-np.sin(beta), 0, np.cos(beta)]])


def get_corner(R, alpha, beta, POINT_REAL):
    gate_center_pos = np.array([[SHIFT_Y], [-0.1469], [-R]])
    point_real_rot = gate_center_pos + (rot_x(alpha*np.pi / 180) @ rot_y(beta*np.pi / 180) @ (POINT_REAL - gate_center_pos))
    # print(rot_x(alpha*np.pi / 180) @ rot_y(beta*np.pi / 180) @ POINT_REAL)
    point_real_rot = np.vstack((point_real_rot, 1))
    c_c = M @ point_real_rot
    corner = c_c[:3] / c_c[3]
    return corner


def generate_label_file(path, alphas, betas, Rs):
    # gi = GeneralImage(path)
    print("Generating corner position for {}".format(path))
    stored_labels = []
    counter = 0
    for R in Rs:
        # Real gate coordinates definition. (0,0,0) - camera

        POINT_REAL1 = np.array([[GATE_WIDTH / 2 - SHIFT_X], [GATE_WIDTH / 2 - SHIFT_Y], [-R]])
        POINT_REAL2 = np.array([[-GATE_WIDTH / 2 - SHIFT_X], [GATE_WIDTH / 2 - SHIFT_Y], [-R]])
        POINT_REAL3 = np.array([[-GATE_WIDTH / 2 - SHIFT_X], [-GATE_WIDTH / 2 - SHIFT_Y], [-R]])
        POINT_REAL4 = np.array([[GATE_WIDTH / 2 - SHIFT_X], [-GATE_WIDTH / 2 - SHIFT_Y], [-R]])
        for alpha in alphas:
            j = 0
            for beta in betas:
                x1, y1, z1 = get_corner(R, alpha, beta, POINT_REAL1)
                x2, y2, z2 = get_corner(R, alpha, beta, POINT_REAL2)
                x3, y3, z3 = get_corner(R, alpha, beta, POINT_REAL3)
                x4, y4, z4 = get_corner(R, alpha, beta, POINT_REAL4)
                xs = np.array([x1, x2, x3, x4,x1])
                ys = np.array([y1, y2, y3, y4,y1])
                print(xs)
                xs = CAMERA_CENTER_PX[0] + xs*CAMERA_RES[0] / (2) # TODO: change name
                ys = CAMERA_CENTER_PX[1] + ys*CAMERA_RES[1] / (2)
                # print(x1-x2)
                stored_labels.append({"Name": "img_{:04d}".format(counter), "R":R, "alpha": alpha, "beta":beta})   #
                for i in range(4):
                    stored_labels[-1]["x{}".format(i)] = xs[i][0]
                    stored_labels[-1]["y{}".format(i)] = ys[i][0]


                if j % 10 == 0:

                    plt.subplot(3,2,j//10+1)
                    # print((j//10+1))
                    plt.plot(xs, ys, color='red')
                    # plt.subplot(6,2,j//6 + 2)
                    gi = GeneralImage(path + '/' + "GateRenders_00{:02d}.jpg".format(j+1))
                    plt.imshow(gi.rgb())
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


def find_axis(path):
    img = None
    ans = None
    for file in os.listdir(path):
        if '.jpg' in file:
            gi = GeneralImage(os.path.join(path,file))
            if img is None:
                img = gi.gray()
                ans = np.ones(img.shape)
            else:
                ans = ans * (img == gi.gray())
    print(np.max(ans))
if __name__ == "__main__":
    path = "../data/cad_renders2"
    alpha = [0]
    beta = np.linspace(0, 90, 60)
    R = [2]
    generate_label_file(path, alpha, beta, R)
    # find_axis(path)
