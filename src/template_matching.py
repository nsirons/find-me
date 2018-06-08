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

    def find_corner(self, img):
        threshold = 0.95
        img_or = np.copy(cv2.imread(img, 0))
        img = np.copy(cv2.imread(img, 0))
        method = eval(self.method)

        res0 = cv2.matchTemplate(img,self.corners[0][0], method)
        res1 = cv2.matchTemplate(img, self.corners[0][1], method)
        res2 = cv2.matchTemplate(img, self.corners[0][2], method)
        res3 = cv2.matchTemplate(img, self.corners[0][3], method)

        min_val0, max_val0, min_loc0, max_loc0 = cv2.minMaxLoc(res0)
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
        min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)

        top_left0 = max_loc0 if max_val0 > threshold else (0, 0)
        top_left1 = max_loc1 if max_val0 > threshold else (0, 0)
        top_left2 = max_loc2 if max_val0 > threshold else (0, 0)
        top_left3 = max_loc3 if max_val0 > threshold else (0, 0)

        bottom_right0 = (top_left0[0] + self.w, top_left0[1] + self.h)
        bottom_right1 = (top_left1[0] + self.w, top_left1[1] + self.h)
        bottom_right2 = (top_left2[0] + self.w, top_left2[1] + self.h)
        bottom_right3 = (top_left3[0] + self.w, top_left3[1] + self.h)

        cv2.rectangle(img, top_left0, bottom_right0, (255, 0, 0), thickness=3)
        cv2.rectangle(img, top_left1, bottom_right1, (255, 0, 0), thickness=3)
        cv2.rectangle(img, top_left2, bottom_right2, (255, 0, 0), thickness=3)
        cv2.rectangle(img, top_left3, bottom_right3, (255, 0, 0), thickness=3)
        # print(max_val0, max_val1, max_val2, max_val3)
        plt.subplot(131), plt.imshow(res0, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(self.method)
        img_can = cv2.Canny(img_or, 100, 200)
        plt.subplot(133)
        plt.imshow(img_can)
        plt.show()
        return None

    def __str__(self):
        return "Template matching: {}".format(self.method)


if __name__ == "__main__":
    path = "../data/cad_renders5"
    tm = TemplateMatching()
    tm.load_corners(os.path.join(path, 'template_corner.png'))
    tm.find_corner("/home/kit/projects/find-me/data/Basement/img_00118.jpg")
    # tm.find_corner("/home/kit/projects/find-me/data/cad_renders2/GateRenders_0010.jpg")
    tm.find_corner("/home/kit/projects/find-me/data/cad_renders_dist/img_3_jpeg_10.jpg")