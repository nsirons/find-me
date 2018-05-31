import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from functools import lru_cache
# TODO: Convert to RGB or HSV?


class CamImage:
    """
    CamImage class reads and converts UYVY image to RGB.
    """
    def __init__(self, path):
        if not os.path.exists(path):
            raise OSError("Image: Invalid path to image".format(path))

        self.__path = path
        self.__w = None
        self.__h = None
        self.__h_per_pixel = None
        self.bytes_per_pixel = None
        self.__data = None
        self.__rgb = None
        self.load()

    def __getitem__(self, px_pos):
        w, h = px_pos
        return self.__rgb[w, h, :]

    def __str__(self):
        return "Image: {}\nWidth={}\tHeight={}\t".format(self.__path, self.__w, self.__h // self.__h_per_pixel)

    def load(self):
        # Assume this standard header
        self.__h, self.__w, self.__h_per_pixel, self.bytes_per_pixel = map(int, np.fromfile(self.__path, dtype=np.uint16, count=4))
        self.__data = np.fromfile(self.__path, dtype=np.uint8)[8::].reshape((self.__w, self.__h))
        Y = self.__data[:, 1::2]
        U = self.__data[:, 0::4]
        V = self.__data[:, 2::4]

        red1, green1, blue1 = self.yuv444torgb888(Y[:, ::2] / 256, U / 256, V / 256)
        red2, green2, blue2 = self.yuv444torgb888(Y[:, 1::2] / 256, U / 256, V / 256)
        self.__rgb = np.zeros((self.__w, self.__h // self.__h_per_pixel, 3), dtype="uint8")
        self.__rgb[:, ::2, 0] = red1
        self.__rgb[:, 1::2,  0] = red2
        self.__rgb[:, ::2,  1] = green1
        self.__rgb[:, 1::2, 1] = green2
        self.__rgb[:, ::2, 2] = blue1
        self.__rgb[:, 1::2, 2] = blue2
        # self.bgr = self.__rgb[:, :, ::-1]  # remove or keep?

    @staticmethod
    def yuv444torgb888(Y, U, V):
        R = np.clip(298.082 * Y               + 408.583 * V - 222.921, 0, 255)
        G = np.clip(298.082 * Y - 100.291 * U - 208.120 * V + 135.576, 0, 255)
        B = np.clip(298.082 * Y + 516.412 * U               - 276.836, 0, 255)
        return R, G, B

    def show_rgb(self):
        plt.imshow(np.rot90(self.__rgb, 3), origin='lowering')
        plt.show()

    def show_orig(self):
        plt.imshow(self.__data[:, 1::2].T, origin="lower", cmap="gray")  # fix
        plt.show()

    @lru_cache(maxsize=None)
    def get_size(self):
        return self.__w, self.__h // self.__h_per_pixel

    @lru_cache(maxsize=None)
    def gray(self):
        return cv2.cvtColor(self.__rgb, cv2.COLOR_RGB2GRAY)

    @lru_cache(maxsize=None)
    def rgb(self):
        return np.copy(self.__rgb)

    @lru_cache(maxsize=None)
    def bgr(self):
        return np.copy(self.__rgb[:,:,::-1])


    # TODO: FIX IT
    def pre_filt(self):
        # print(self.__rgb[60,50])
        # cv2.imwrite("test.jpg", self.bgr)
        self.hsv = cv2.cvtColor(self.__rgb, cv2.COLOR_RGB2HSV)
        orange_low = np.array([165, 10, 10], dtype="uint8")
        orange_high = np.array([179, 255, 255], dtype="uint8")
        # orange = np.array([160,160, 160], dtype="uint8")
        # mask = cv2.inRange(self.__rgb, orange-40, orange+40)  # TODO : set threshold for color filtering
        self.mask = cv2.inRange(self.hsv, orange_low, orange_high)
        # print(np.max(mask))
        res = cv2.bitwise_and(self.__rgb, self.__rgb, mask=self.mask)
        # res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        # print(res)

        # plt.imshow(self.mask, cmap='gray')
        plt.imshow(res)
        # print(self.__rgb[0,0])
        plt.show()


if __name__ == "__main__":
    img = CamImage("data/qcif90/img_00020.raw")
    img.show_rgb()
    img.pre_filt()