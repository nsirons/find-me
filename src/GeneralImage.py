import cv2
from functools import lru_cache
import matplotlib.pyplot as plt


class GeneralImage:
    def __init__(self, path):
        self.__img = cv2.imread(path)

    @lru_cache(maxsize=None)
    def get_size(self):
        return self.__img.shape

    @lru_cache(maxsize=None)
    def rgb(self):
        return self.__img[:, :, ::-1]

    @lru_cache(maxsize=None)
    def bgr(self):
        return self.__img

    @lru_cache(maxsize=None)
    def gray(self):
        return cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)

    @lru_cache(maxsize=None)
    def hsv(self):
        return cv2.cvtColor(self.__img, cv2.COLOR_BGR2HSV)

    def color_stats(self, cmap='rgb', rect=(0, 0, 100, 150)):  # TODO: Fix for gray (if needed)
        labels = {'bgr': ('blue', 'green', 'red'), 'rgb': ('red', 'green', 'blue'),
                  'gray': ('gray',), 'hsv': ('hue', 'saturation', 'v')}
        colors = {'bgr': ('b', 'g', 'r'), 'rgb': ('r', 'g', 'b'),
                  'gray': ('gray',), 'hsv': ('c', 'm', 'y')}
        x, y, w, h = rect
        img = getattr(self, cmap)()[x:x + w, y:y + h, :]
        for i in range(3):
            plt.subplot(241 + i)
            plt.imshow(img[:, :, i], cmap=cmap if 'hsv' == cmap else None)

        plt.subplot(244)
        plt.imshow(img, cmap=cmap if 'hsv' == cmap else None)
        plt.subplot(212)
        for i, label in enumerate(labels[cmap]):
            plt.hist(img[:, :, i].flatten(), bins=50, label=label, color=colors[cmap][i], histtype="step")
        plt.legend()
        plt.show()

    def show_img(self):
        cv2.imshow('image', self.__img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # TODO: remove later, used for get information about edges on the image
    def edge_stats(self):
        laplacian = cv2.Laplacian(self.gray(), cv2.CV_64F)
        sobelx = cv2.Sobel(self.gray(), cv2.CV_64F, 1, 0, ksize=19)
        sobely = cv2.Sobel(self.gray(), cv2.CV_64F, 0, 1, ksize=19)
        plt.subplot(2, 2, 1), plt.imshow(self.__img, cmap='gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
        plt.plot()
        # cv2.Canny(self.__img,)

if __name__ == '__main__':
    path = "data/Cyberzoo/img_00000.jpg"
    gi = GeneralImage(path)
    # gi.color_stats('rgb', (150, 100, 50, 50))
    gi.edge_stats()
    # gi.show_img()
