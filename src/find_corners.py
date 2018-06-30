import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from src.Renamed__img import ImagePreProc
from shutil import copyfile

import csv
# TODO: Be able tochange Camera Center, shifts and rotations
# http://www.songho.ca/opengl/gl_projectionmatrix.html - Clear explanation of Perspective Frustum
# Gate information

GATE_WIDTH = 1.4
FOV = 105 /2 * np.pi / 180


# TODO : Change parameters (n,f to better values?)
# Camera planes parameters
n = 24.3 ** 10**-3  # Near plane
t = np.tan(FOV) * n  # right
r = 512 / 288 * t  # top
f = 10  # Far plane
# scale = 1 / np.tan(FOV /2)
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
    point_real_rot = np.vstack((point_real_rot * np.tan(FOV/2)*R * 2 / 512, 1))
    # print(point_real_rot)
    c_c = M @ point_real_rot
    # print(c_c)
    corner = c_c[:3] / c_c[3]
    return corner


def generate_label_file(path, ext, with_gate,*,camera_res,shift_y, shift_x):
    CAMERA_RES = camera_res
    CAMERA_CENTER_PX = np.array([[CAMERA_RES[0] // 2], [CAMERA_RES[1] // 2], [0]])
    # SHIFT_Y = 0.15523613963039012
    # SHIFT_Y = -0.15
    # SHIFT_X = 0.081451
    SHIFT_Y = shift_y
    SHIFT_X = shift_x
    # gi = GeneralImage(path)
    print("Generating corner position for {}".format(path))
    stored_labels = []
    counter = 1
    img_lab = ImagePreProc()
    img_lab.load(path, ext, not bool(with_gate))
    selected = np.random.randint(0, len(img_lab), 9)
    for key in img_lab.get_images():
        if with_gate:
            R = float(img_lab[key]['R'])
            alpha = float(img_lab[key]['alpha'])
            beta = float(img_lab[key]['beta'])
            SHIFT_X = shift_x*np.sin(beta*np.pi/180*2)
            POINT_REAL1 = np.array([[GATE_WIDTH / 2 + SHIFT_X], [GATE_WIDTH / 2 + SHIFT_Y], [-R]])
            POINT_REAL2 = np.array([[-GATE_WIDTH / 2 + SHIFT_X], [GATE_WIDTH / 2 + SHIFT_Y], [-R]])
            POINT_REAL3 = np.array([[-GATE_WIDTH / 2 + SHIFT_X], [-GATE_WIDTH / 2 + SHIFT_Y], [-R]])
            POINT_REAL4 = np.array([[GATE_WIDTH / 2 + SHIFT_X], [-GATE_WIDTH / 2 + SHIFT_Y], [-R]])

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
            xs = 1280/512*np.array([x1, x4, x3, x2, x1])
            ys = 1280/512*np.array([y1, y4, y3, y2, y1])

            xs = CAMERA_CENTER_PX[0] + xs*CAMERA_RES[0] / 2 # TODO: change name
            ys = CAMERA_CENTER_PX[1] + ys*CAMERA_RES[1] / 2

            stored_labels.append({"Name": img_lab[key]['name'], "R": R, "alpha": alpha, "beta":beta, "gate":int(with_gate)})   # TODO: change gate status for empty pictures
            for i in range(4):
                stored_labels[-1]["x{}".format(i)] = xs[i][0]
                stored_labels[-1]["y{}".format(i)] = ys[i][0]

            if key in selected:

                plt.subplot(3, 3, counter)

                plt.plot(xs, ys, color='red',label=R)
                plt.legend()
                font = cv2.FONT_HERSHEY_SIMPLEX
                img = cv2.imread(os.path.join(path, img_lab[key]["name"]))
                cv2.putText(img, 'c0', (xs[0], ys[0]), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'c1', (xs[1], ys[1]), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'c2', (xs[2], ys[2]), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'c3', (xs[3], ys[3]), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                # cv2.putText(img, str(beta), (10,200), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                plt.imshow(img)
                counter += 1

        else:
            stored_labels.append({"Name": img_lab[key]['name'], "R": 0, "alpha": 0, "beta":0, "gate": int(with_gate)})   # TODO: change gate status for empty pictures
            for i in range(4):
                stored_labels[-1]["x{}".format(i)] = 0
                stored_labels[-1]["y{}".format(i)] = 0

    print("Storing data to: {}".format("{}/{}".format(path, "corners.txt")))
    with open("{}/{}".format(path, "corners.txt"), mode='w+') as csvfile:
        fieldnames = stored_labels[-1].keys()
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        csvwriter.writeheader()
        for label in stored_labels:
            csvwriter.writerow(label)
    plt.show()


def copy_images(from_path, to_path):
    store_stash = []
    import random
    files = random.choices(os.listdir(from_path), k=15)
    for file_name in files:
        if file_name[-4:] != ".txt":
            # copyfile(os.path.join(from_path, file_name), os.path.join(to_path, file_name))
            img = cv2.imread(os.path.join(from_path, file_name))
            img = cv2.resize(img, (512, 288))
            cv2.imwrite(os.path.join(to_path, file_name), img)
        else:
            with open(os.path.join(from_path, file_name), mode='r') as csvfile:
                reader = csv.DictReader(csvfile, dialect="excel-tab")
                for row in reader:
                    store_stash.append(row)
            # print(store_stash)
            with open(os.path.join(to_path, file_name), mode="a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=store_stash[0].keys(), dialect="excel-tab")
                for line in store_stash:
                    writer.writerow(line)

if __name__ == "__main__":
    bg_path = "../data/bg_less"
    path_valid_35 = "../data/validation/3-5"
    path_valid_360 = "../data/validation/3-60"
    path_valid_53 = "..data/validation/5-3"
    path_valid_560 = "../data/validation/5-60"
    # for i in (2,3,5):
        # path = "../data/cad_renders{}_dist".format(i)
    # path = "../data/backgrounds"
    #     generate_label_file(path, '.jpg', True)
    # path = '../data/validation/5-3'
    # generate_label_file(path, '.jpg', True, camera_res=(512, 288), shift_y=-0.0, shift_x=-np.tan(FOV)*2*20./512)  # by radius, and around 20px
    # generate_label_file(bg_path, '.jpg', False)
    generate_label_file(bg_path, '.jpg', False, camera_res=(512, 288), shift_y=-0.0, shift_x=-np.tan(FOV) * 2 * 20. / 512)
    copy_images(bg_path, path_valid_35)
    copy_images(bg_path, path_valid_360)
    copy_images(bg_path, path_valid_53)
    copy_images(bg_path, path_valid_560)

