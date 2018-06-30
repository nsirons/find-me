import numpy as np
import cv2
import os
from src.benchmark import Benchmark
from copy import deepcopy

import matplotlib.pyplot as plt

class TemplateMatching:
    """
    Applies template matching to detect gate on image
    """
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    def __init__(self):
        self.images = None
        self.corners = None
        self.w = None
        self.h = None
        self.transformation = None
        self.pts1 = None

    def load_corners(self, path):
        if self.corners is None:
            self.corners = {}
        corner = cv2.imread(os.path.join(path, "template_orange.png"))

        corner_normal = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        self.w, self.h = corner_normal.shape
        # tl = cv2.imread(os.path.join(path, "top_left_4.png"),  0)
        # tr = cv2.imread(os.path.join(path, "top_right_2.png"), 0)

        # example = [tl, tr, np.flip(tr, axis=0), np.flip(tl, axis=0)]
        self.corners["nominal"] = [np.rot90(corner_normal)]  # TODO: small mistake
        # self.corners["top_left"] = [example]
        for i in range(3):
            self.corners["nominal"].append(np.rot90(self.corners["nominal"][i]))
            # self.corners["top_left"].append([np.rot90(self.corners["top_left"][i][j-1], axes=(1, 0)) for j in range(4)])

        # if corner_normal.shape == tl.shape == tr.shape:
        #     self.w, self.h = corner_normal.shape
        # else:
        #     raise IndexError("Corners of templates have different sizes")

        center_x, center_y, r = (8, 11, 7)
        pts1 = np.float32([[center_x, center_y-r], [center_x, center_y], [center_x+r, center_y]])
        transformation = lambda alpha: np.float32([[center_x, center_y - r],
                                                       [center_x, center_y],
                                                       [center_x + r * np.cos(np.pi / 180 * alpha), center_y - r * np.sin(np.pi / 180*alpha)]])

        # generate new rotated templates
        for new_angle in np.arange(-60, 61, 5):
            pts2 = transformation(new_angle)
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(corner_normal, M, (16,16))
            # averages = [np.mean(dst[:, i][dst[:, i] > 25], axis=0) for i in range(dst.shape[0])]

            # fill empty space
            for i in range(dst.shape[0]):
                # col = dst[:, i]
                # col[np.where(col < 25)[0]] = averages[i]
                # dst[:, i] = col
                self.corners[new_angle] = dst
            # self.corners[new_angle] = cv2.dilate(dst[int(7*np.sin(abs(new_angle)*np.pi/180)/2):int(19-7*np.sin(abs(new_angle)*np.pi/180) /2),
            #                                                   int(2+7*np.sin(abs(new_angle)*np.pi/180)/2):int(19-7*np.sin(abs(new_angle)*np.pi/180) /2)], (3,3), iterations=2)
            # self.corners[new_angle] = cv2.cvtColor(dst[int(7*np.sin(abs(new_angle)*np.pi/180)/2):int(19-7*np.sin(abs(new_angle)*np.pi/180) /2),
            #                                                   int(2+7*np.sin(abs(new_angle)*np.pi/180)/2):int(19-7*np.sin(abs(new_angle)*np.pi/180) /2)],
            #                                                   cv2.COLOR_BGR2GRAY)
            # plt.imshow(self.corners[new_angle])
            # plt.show()
        # print(self.)
        for i, angle in enumerate([0,30,-30]):
            plt.subplot(131+i)
            plt.imshow(self.corners[angle], 'gray')
        # plt.show()

    def test_single(self, path, method, threshold):
        if method not in self.methods:
            raise ValueError("Invalid method called: {}".format(method))
        if threshold < 0 or threshold > 1:
            raise ValueError("Invalid threshold, should be in range of 0 to 1 : {}".format(threshold))
        cls = deepcopy(self)
        # cls.threshold = threshold
        cls.method = eval(method)
        b = Benchmark(cls)
        b.load_data(path)
        return b.test(True)

    # TODO: Implement method/threshold calibration
    def test_many(self, path, methods=None, thresholds=None, save_name="TM_results.csv"):

        methods = self.methods if methods is None else methods
        thresholds = np.arange(0.945, 0.985, 0.5) if thresholds is None else thresholds
        b = Benchmark(deepcopy(self))
        b.load_data(path)
        data = b.test_batch(methods, thresholds, name="Template Matching")
        # for method in methods:
        #     for threshold in thresholds:
        #         res = self.test_single(path, method, threshold)
        #         data.append(res)

        # TODO: For now save data
        import csv
        with open(save_name, mode='w') as csvfile:
            fieldnames = ["gate"]
            for s in ['mean', 'max', 'min']:
                for p in range(4):
                    for coord in ["x", "y"]:
                        fieldnames.append("{}_{}_{}".format(s,coord,p))

            csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="\t")
            csvwriter.writeheader()
            conv = {'x':0, 'y':1}
            for stat in data:
                row = {"gate": stat["gate"]}
                for s in ['mean', 'max', 'min']:
                    for p in range(4):
                        for coord in ['x', 'y']:
                            row["{}_{}_{}".format(s, coord, p)] = stat["error"][s][conv[coord]*4+p]

                csvwriter.writerow(row)

    def find_gate(self, img_name, *args):
        method, threshold = args
        method = eval(method)
        img = np.copy(cv2.imread(img_name, 0))
        corners = self.find_corners(img, self.corners["nominal"], method, threshold)
        gate_detected = self.find_gate_detected(corners)

        # TODO: Does 1 extra template matching for curved corners (1st time)
        # TODO: Smarter use of TM (if 2 points are far away, or not feasible, stop doing TM)
        # if not gate_detected:
        #     # corner_selection = self.find_corners(img, map(lambda x: x[0], self.corners["top_left"]), method, threshold)
        #     curved_corners = [self.corners[i] for i in np.arange(0, 60, 5)]
        #     corner_selection = self.find_corners(img, curved_corners, method, threshold, True)
        #     selected = sorted(corner_selection, key=lambda x: x["max_val"])[-1]  # selected template by maximum value
        #     # selected_index = [i for i in corner_selection if i == selected]
        #     if selected["max_val"] > threshold:
        #         # continue with other 3 corners
        #         # corners = self.find_corners(img, self.corners["top_left"][selected["id"]], method, threshold)
        #         f_angle = selected["angle"]
        #         print(f_angle)
        #         other_3 = [np.flip(self.corners[f_angle], axis=0),
        #                    np.flip(self.corners[-abs(f_angle)], axis=1),
        #                    np.flip(np.flip(self.corners[-abs(f_angle)], axis=1), axis=0)]
        #         # for i in range(4):
        #         #     if i == 0:
        #         #         plt.subplot(221)
        #         #         plt.imshow(self.corners[f_angle], cmap="gray")
        #         #     else:
        #         #         plt.subplot(221+i)
        #         #         plt.imshow(other_3[i-1], cmap="gray")
        #         # plt.show()
        #         corners = self.find_corners(img, other_3, method, threshold)
        #         # Change order of corners, so it matches order of benchmark
        #         corners = [corners[1], corners[2], corners[0], selected]
        #         # Re-check obtained corners to gate
        #         gate_detected = self.find_gate_detected(corners)

        return gate_detected, corners[0]["max_loc"], corners[1]["max_loc"], corners[2]["max_loc"], corners[3]["max_loc"]

    def find_corners(self, img, templates, method, threshold, angle=False):
        corners = []
        for i, template in enumerate(templates):
            try:
                res = cv2.matchTemplate(img, template, method)
            except TypeError:
                raise TypeError("Invalid img ", img)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc if max_val > threshold else (0, 0)
            bottom_right = (top_left[0] + self.w, top_left[1] + self.h)
            corners.append({"id": i, "det": top_left != (0, 0), "res": res, "max_val": max_val, "max_loc": max_loc,
                            "top_left": top_left, "bot_right": bottom_right, "angle": 0 if not angle else 0+i*5})  # TODO: change hardcode
        # print(i)
        return corners

    def find_gate_detected(self, corners):
        if [corner["det"] for corner in corners].count(True) > 2:
            # check order of detected corners:
            if self.corners_ordered(corners):
                if self.corners_dont_cross(corners):
                    if self.check_shape(corners):
                        if self.check_scale(corners):
                            return True
        return False

    @staticmethod
    def corners_ordered(corners, start=0):
        # check = {0: (0,-1), 1: (1, 1), 2: (0, 1), 3: (1, -1)}
        cur = start
        for i, corner in enumerate(corners[1:]):
            if (corner["max_loc"][(cur+1) % 2] - corners[i]["max_loc"][(cur+1) % 2])*(-1+2*((cur) // 2)) < 0:
                return False
            cur += 1
        return True

    @staticmethod
    def corners_dont_cross(corners):
        if corners[0]["max_loc"][0] > corners[2]["max_loc"][0]:
            if corners[0]["max_loc"][1] > corners[2]["max_loc"][1]:
                if corners[1]["max_loc"][0] > corners[3]["max_loc"][0]:
                    if corners[1]["max_loc"][1] < corners[3]["max_loc"][1]:
                        return True
        return False

    @staticmethod
    def check_shape(corners, threshold=20):
        if abs(corners[0]["max_loc"][0]-corners[1]["max_loc"][0]) < threshold:
            if abs(corners[0]["max_loc"][1] - corners[3]["max_loc"][1]) < threshold:
                return True
            elif abs(corners[1]["max_loc"][1] - corners[2]["max_loc"][1]) < threshold:
                return True
        if abs(corners[2]["max_loc"][0] - corners[3]["max_loc"][0]) < threshold:
            if abs(corners[2]["max_loc"][1] - corners[1]["max_loc"][1]) < threshold:
                return True
            elif abs(corners[3]["max_loc"][1] - corners[0]["max_loc"][1]) < threshold:
                return True
        return False

    @staticmethod
    def check_scale(corners):
        d = [np.linalg.norm(np.array(corners[i % 4]["max_loc"]) - np.array(corners[(i+1) % 4]["max_loc"])) for i in range(4)]
        f = 0
        for i in range(4):
            if not (0.5 < d[i % 4]/d[(i+1) % 4] < 1.5):
                f += 1
        return True if f < 2 else False

    def animation(self, path):
        print("Rendering images")
        images = []

        for filename in sorted(os.listdir(path)):
            if '.jpg' in filename:
                path_to_file = os.path.join(path, filename)
                img = cv2.imread(path_to_file)
                gd, c0, c1, c2, c3 = self.find_gate(path_to_file, "cv2.TM_CCORR_NORMED", 0.95)

                cv2.rectangle(img, (c0[0]+self.w,) + (c0[1]+self.h,),
                                   (c0[0]-self.w,) + (c0[1]-self.h,), (255, 0, 0), thickness=3)
                cv2.rectangle(img, (c1[0]+self.w,) + (c1[1]+self.h,),
                                   (c1[0]-self.w,) + (c1[1]-self.h,), (255, 0, 0), thickness=3)
                cv2.rectangle(img, (c2[0]+self.w,) + (c2[1]+self.h,),
                                   (c2[0]-self.w,) + (c2[1]-self.h,), (255, 0, 0), thickness=3)
                cv2.rectangle(img, (c3[0]+self.w,) + (c3[1]+self.h,),
                                   (c3[0]-self.w,) + (c3[1]-self.h,), (255, 0, 0), thickness=3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'c0', c0, font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'c1', c1, font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'c2', c2, font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'c3', c3, font, 4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, str(gd), (50, 100), font, 4, (255, 0, 0), 2, cv2.LINE_AA)
                images.append(img)


        print("Rendering Done")
        i = 0
        while True:
            k = cv2.waitKey(3000) & 0xff
            cv2.imshow("Template matching animation", images[i % len(images)])
            if k == 27:  # wait for ESC key to exit
                break
            i += 1
        cv2.destroyAllWindows()

    def __str__(self):
        return "Template matching: {}".format("NONNOENONEO")


if __name__ == "__main__":
    path = "../data/cad_renders3_new"
    path = "/home/kit/projects/find-me/data/20June2/GateRenders360Rotation/GateRenders360Rotation_animation"
    path_corners = "../data/corners"
    tm = TemplateMatching()
    tm.load_corners(path_corners)
    times = []
    import time
    import random
    files = random.choices(os.listdir("../data/validation/5-3"), k=10)

    # 0.05964167339843698 -> 16.766799840096485 Hz
    for i in range(10):
        t1 = time.perf_counter()
        print(files[i])
        tm.find_gate(os.path.join("../data/validation/5-3", files[i]), "cv2.TM_CCORR_NORMED", 0.95)
        t = time.perf_counter() - t1
        times.append(t)
    print(times)
    print(np.mean(times))


    # tm.animation(path)
    # # tm.test_many(path, methods=["cv2.TM_CCOEFF_NORMED"])
    # tm.animation(path, True)
    # cv2.warpPerspective()
    # euler = lambda point, alpha: [point[0]*np.cos(alpha), point[1]*np.sin(alpha)]
    # import matplotlib.pyplot as plt
    # # func = lambda alpha: np.float32([[8-7*np.sin(np.pi/180*alpha), 11-7*np.cos(np.pi/180*alpha)],
    # #                                  [8, 11],
    # #                                  [8+7*np.cos(np.pi/180*alpha), 11-7*np.sin(np.pi/180*alpha)]])
    #
    # func = lambda alpha: np.float32([[8-0*np.sin(np.pi/180*alpha), 4-0*np.cos(np.pi/180*alpha)],
    #                                  [8, 11],
    #                                  [8+7*np.cos(np.pi/180*alpha), 11-7*np.sin(np.pi/180*alpha)]])
    # #
    # img = cv2.imread(os.path.join(path_corners, "template_corner.png"))
    # rows, cols, ch = img.shape
    #
    # pts1 = np.float32([[8, 4], [8, 11], [15, 11]])
    # pts2 = func(45)
    #
    # M = cv2.getAffineTransform(pts1, pts2)
    #
    # dst = cv2.warpAffine(img, M, (20, 20))
    #
    # averages = [np.mean(dst[:, i, :], axis=0) for i in range(dst.shape[0])]
    #
    # for i in range(dst.shape[0]):
    #     col = dst[:, i,:]
    #     col[np.where(col < (25,25,25))[0]] = averages[i]
    #     dst[:, i, :] = col
    #     # print(col == (0,0,0))
    #
    # plt.subplot(133), plt.imshow(dst[:, 1:18, :]) # TODO : figure out
    #
    # # dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, (15, 15), iterations=3)
    # # dst = cv2.GaussianBlur(dst, (5,5), sigmaX=10, sigmaY=10)
    # # dst = cv2.dilate(dst, (5, 5), iterations=2)
    # plt.subplot(131), plt.imshow(img), plt.plot([i[0] for i in pts1], [i[1] for i in pts1]), plt.title('Input')
    # plt.subplot(132), plt.imshow(dst), plt.plot([i[0] for i in pts2], [i[1] for i in pts2]), plt.title('Output')
    # # print(dst)
    # plt.show()