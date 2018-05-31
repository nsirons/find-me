from CamImage import CamImage  # fix this error?
import cv2
import numpy as np
import matplotlib.pyplot as plt


class FeatureBasedTest:
    """
    FeatureBasedTest class is used to test different feature based approaches to detect gates
    """
    def __init__(self, path):
        self.img = CamImage(path)


    # TODO: for harris and shi cut the image at the region where the gate is present?
    # TODO: Remove/blur the lights?
    def harris_corner(self):
        img = self.img.bgr()
        gray = self.img.gray()
        dst = cv2.cornerHarris(gray, 2,3, 0.01) # change parameters
        dst = cv2.dilate(dst, None)
        img[dst>0.01*dst.max()] = [0,0,255]

        cv2.imshow('dst', img)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    def shi_tomasi_corner(self):
        gray = self.img.gray()
        img = self.img.rgb()
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.5, 20)
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 3, 255, -1)

        plt.imshow(img), plt.show()

    def sift(self):

        img = self.img.rgb()
        gray = self.img.gray()

        sift = cv2.xfeatures2d.SIFT_create()
        # sift = cv2.Feature2D
        kp = sift.detect(gray, None)
        img = cv2.drawKeypoints(gray, kp, img)

        cv2.imshow('dst', img)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    def surf(self):
        # img = self.img.rgb()
        gray = self.img.gray()

        surf = cv2.xfeatures2d.SURF_create(100)
        # kp, des = surf.detectAndCompute(gray, None)
        surf.setHessianThreshold(200,   300)
        kp, des = surf.detectAndCompute(gray, None)
        img2 = cv2.circle(gray, kp, None, (255, 0, 0), 4)
        plt.imshow(img2, cmap='gray'), plt.show()

    def canny_edge(self):
        edges = cv2.Canny(self.img.gray(), 25, 200)
        plt.imshow(edges)
        plt.show()

if __name__ == "__main__":
    path = "data/qcif90/img_00020.raw"
    fbt = FeatureBasedTest(path)
    # fbt.harris_corner()
    # print(cv2.xfeatures2d)
    # fbt.shi_tomasi_corner()
    # fbt.sift()
    # fbt.surf()
    fbt.canny_edge()