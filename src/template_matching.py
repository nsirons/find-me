import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


class TemplateMatching:
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    def __init__(self):
        self.method = 'cv2.TM_CCORR_NORMED'
        self.images = None
        self.corners = None
        self.w = None
        self.h = None

    def load_corners(self, path):
        # TODO: change, key and corner dict, different angles?
        if self.corners is None:
            self.corners = {}
        corner = cv2.imread(path, 0)
        self.corners = {}
        self.corners[0] = [i for i in range(4)]
        for i in range(4):
            self.corners[0][i] = np.rot90(corner) if i == 0 else np.rot90(self.corners[0][i-1])

        self.w, self.h = corner.shape

    def load_images(self, path):
        self.images = []
        for file_name in sorted(os.listdir(path)):
            if '.jpg' in file_name:
                self.images.append(cv2.imread(os.path.join(path, file_name), 0))

    def test_all(self, path):
        self.load_images(path)  # load dataset

    def test_current(self, path):
        self.load_images(path)

    def find_gate(self, img):
        threshold = 0.95
        img_or = np.copy(cv2.imread(img, 0))
        img = np.copy(cv2.imread(img, 0))
        method = eval(self.method)
        # res = []
        corners = []
        for i in range(4):
            res = cv2.matchTemplate(img, self.corners[0][i], method)
            # cv2.imshow("Image", self.corners[0][0])
            # while True:
            #     k = cv2.waitKey(0)
            #     if k == 27:  # wait for ESC key to exit
            #         cv2.destroyAllWindows()

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc if max_val > threshold else (0, 0)
            bottom_right = (top_left[0] + self.w, top_left[1] + self.h)
            corners.append({"det": top_left != (0,0), "res": res, "max_val":max_val, "max_loc": max_loc,
                            "top_left":top_left, "bot_right":bottom_right})

        # min_val0, max_val0, min_loc0, max_loc0 = cv2.minMaxLoc(res0)
        # min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        # min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
        # min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)

        # top_left0 = max_loc0 if max_val0 > threshold else (0, 0)
        # top_left1 = max_loc1 if max_val0 > threshold else (0, 0)
        # top_left2 = max_loc2 if max_val0 > threshold else (0, 0)
        # top_left3 = max_loc3 if max_val0 > threshold else (0, 0)
        #
        # bottom_right0 = (top_left0[0] + self.w, top_left0[1] + self.h)
        # bottom_right1 = (top_left1[0] + self.w, top_left1[1] + self.h)
        # bottom_right2 = (top_left2[0] + self.w, top_left2[1] + self.h)
        # bottom_right3 = (top_left3[0] + self.w, top_left3[1] + self.h)

        cv2.rectangle(img, corners[0]["top_left"], corners[0]["bot_right"], (255, 0, 0), thickness=3)
        cv2.rectangle(img, corners[1]["top_left"], corners[1]["bot_right"], (255, 0, 0), thickness=3)
        cv2.rectangle(img, corners[2]["top_left"], corners[2]["bot_right"], (255, 0, 0), thickness=3)
        cv2.rectangle(img, corners[3]["top_left"], corners[3]["bot_right"], (255, 0, 0), thickness=3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'c0', corners[0]["bot_right"], font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'c1', corners[1]["bot_right"], font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'c2', corners[2]["bot_right"], font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'c3', corners[3]["bot_right"], font, 4, (255, 255, 255), 2, cv2.LINE_AA)

        gate_detected = False
        # check number of detecton (should be more than 3 corners)
        if [corner["det"] for corner in corners].count(True) > 2:
            # check order of detected corners:
            if self.corners_ordered(corners):
                if self.corners_dont_cross(corners):
                    gate_detected = True
        cv2.putText(img, str(gate_detected), (50,100), font, 4, (255, 0, 0), 2, cv2.LINE_AA)
        # print(max_val0, max_val1, max_val2, max_val3)
        # plt.subplot(131), plt.imshow(res, cmap='gray')
        # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        # plt.subplot(132), plt.imshow(img, cmap='gray')
        # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.suptitle(self.method)
        # img_can = cv2.Canny(img_or, 100, 200)
        # plt.subplot(133)
        # plt.imshow(img_can)
        # plt.show()
        # print(gate_detected)
        # return (gate_detected, img)

        return gate_detected, corners[0]["max_loc"], corners[1]["max_loc"], corners[2]["max_loc"], corners[3]["max_loc"]

    @staticmethod
    def corners_ordered(corners, start=0):
        # check = {0: (0,-1), 1: (1, 1), 2: (0, 1), 3: (1, -1)}
        cur = start
        ordered = True

        for i, corner in enumerate(corners[1:]):
            if (corner["max_loc"][(cur+1) % 2] - corners[i]["max_loc"][(cur+1) % 2])*(-1+2*((cur) // 2)) < 0:
                # print((corner["max_loc"][(cur+1) % 2] - corners[i]["max_loc"][(cur+1) % 2])*(1-2*((cur+1) % 2)), cur)
                return False
            cur += 1
            # print(cur)
        return ordered

    @staticmethod
    def corners_dont_cross(corners):
        if corners[0]["max_loc"][0] > corners[2]["max_loc"][0]:
            if corners[0]["max_loc"][1] > corners[2]["max_loc"][1]:
                if corners[1]["max_loc"][0] > corners[3]["max_loc"][0]:
                    if corners[1]["max_loc"][1] < corners[3]["max_loc"][1]:
                        return True
        return False

    def __str__(self):
        return "Template matching: {}".format(self.method)


if __name__ == "__main__":
    path = "../data/cad_renders5"
    tm = TemplateMatching()
    tm.load_corners(os.path.join(path, 'template_corner.png'))
    # tm.find_corner("/home/kit/projects/find-me/data/Basement/img_00118.jpg")
    # tm.find_corner("/home/kit/projects/find-me/data/cad_renders2/GateRenders_0010.jpg")
    # img = []
    # detected = []
    # for i in range(1, 30):
    #     # print("/home/kit/projects/find-me/data/cad_renders2/GateRenders_{:04d}.jpg".format(i))
    #     gd, im = tm.find_gate("/home/kit/projects/find-me/data/cad_renders2/GateRenders_{:04d}.jpg".format(i))
    #     detected.append(gd)
    #     img.append(im)
    # print(img[0])
    # i = 0
    # while True:
    #     k = cv2.waitKey(30) & 0xff
    #     i += 1
    #     # print(i)
    #     cv2.imshow("Rotating    ", img[i % len(detected)])
    #     if k == 27:  # wait for ESC key to exit
    #         break
    # cv2.destroyAllWindows()
    from src.benchmark import Benchmark
    b = Benchmark(tm)
    path = "../data/cad_renders2"
    b.load_data(path)
    b.test()
    print(b)