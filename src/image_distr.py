from PIL import Image
import cv2
import os
import numpy as np
from src.GeneralImage import GeneralImage
import matplotlib.pyplot as plt


# TODO: add more https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# TODO: add contasrt
class DistortImages:
    def __init__(self):
        self.__path = None
        self.images = None

    def load(self, path, ext):
        if self.__path is None:
            self.__path = path
        self.images = []
        for file_name in filter(lambda x: x[-len(ext):] == ext, os.listdir(self.__path)):
            self.images.append((file_name, cv2.imread(os.path.join(path, file_name))))

    def distort(self, save_path, **kwargs):
        # Create folder
        if self.__path is None:
            raise ValueError("Images are not being loaded")

        if not os.path.exists(save_path):
            print("New folder created: {}".format(save_path))
            os.makedirs(save_path)
        print("Deleting images in: {}".format(save_path))

        #  Clean folder
        for file_delete in os.listdir(save_path):
            file_path = os.path.join(save_path, file_delete)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        print("Generating new images")

        for img_name, img in self.images:
            for func in kwargs:
                if func != "compression":
                    self.__class__.__dict__[func].__func__(img_name, img, save_path, kwargs[func])
                else:
                    self.compression(img_name, save_path, kwargs[func])

    @staticmethod
    def blur(file_name, img, save_path, sigma_values, kernel_size=(5,5)):  # TODO: change kernel size?
        for sigma_value in sigma_values:
            new_name = file_name[:-4] + '_' + "blur{:03d}".format(sigma_value) + file_name[-4:]
            cv2.imwrite(os.path.join(save_path, new_name), cv2.GaussianBlur(img, kernel_size, sigmaX=sigma_value, sigmaY=sigma_value))

    # Compression is special, because it requires to load image directly from file, hopefully it does not break anything
    def compression(self, file_name, save_path, quality_values):
        if min(quality_values) <= 0 or max(quality_values) > 100:
            raise ValueError("Invalid quality parameter, not in range of 0-100: {}-{}".format(min(quality_values),
                                                                                              max(quality_values)))
        im = Image.open(os.path.join(self.__path, file_name))
        for quality_val in quality_values:
            new_name = file_name[:-4] + '_' + "compr{:03d}".format(quality_val) + file_name[-4:]
            im.save(os.path.join(save_path, new_name), format='JPEG', subsampling=0, quality=quality_val)

    @staticmethod
    def gaussian_noise(file_name, img, save_path, sigma_values, mu=0):
        for sigma_value in sigma_values:
            new_name = file_name[:-4] + '_' + "noise{:03d}".format(sigma_value) + file_name[-4:]
            cv2.imwrite(os.path.join(save_path, new_name), np.clip(img + np.random.normal(mu, sigma_value, img.shape).astype(np.uint8), 0, 255))

    @staticmethod
    def erosion(file_name, img, save_path, iter_values, kernel_size=(5,5)):
        kernel = np.ones(kernel_size, np.uint8)
        for iter_value in iter_values:
            new_name = file_name[:-4] + '_' + "erosion{:03d}".format(iter_value) + file_name[-4:]
            cv2.imwrite(os.path.join(save_path, new_name), cv2.erode(img, kernel, iterations=iter_value))

    @staticmethod
    def dilation(file_name, img, save_path, iter_values, kernel_size=(5,5)):
        kernel = np.ones(kernel_size, np.uint8)
        for iter_value in iter_values:
            new_name = file_name[:-4] + '_' + "dilation{:03d}".format(iter_value) + file_name[-4:]
            cv2.imwrite(os.path.join(save_path, new_name), cv2.dilate(img, kernel, iterations=iter_value))

    @staticmethod
    def closing(file_name, img, save_path, iter_values, kernel_size=(5,5)):
        kernel = np.ones(kernel_size, np.uint8)
        for iter_value in iter_values:
            new_name = file_name[:-4] + '_' + "closing{:03d}".format(iter_value) + file_name[-4:]
            cv2.imwrite(os.path.join(save_path, new_name), cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iter_value))

    @staticmethod
    def opening(file_name, img, save_path, iter_values, kernel_size=(5,5)):
        kernel = np.ones(kernel_size, np.uint8)
        for iter_value in iter_values:
            new_name = file_name[:-4] + '_' + "opening{:03d}".format(iter_value) + file_name[-4:]
            cv2.imwrite(os.path.join(save_path, new_name), cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iter_value))

    @staticmethod
    def gradient(file_name, img, save_path, iter_values, kernel_size=(5,5)):
        kernel = np.ones(kernel_size, np.uint8)
        for iter_value in iter_values:
            new_name = file_name[:-4] + '_' + "gradient{:03d}".format(iter_value) + file_name[-4:]
            cv2.imwrite(os.path.join(save_path, new_name), cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=iter_value))

    def __str__(self):
        if self.__path is None:
            return "Nothing is loaded"
        return "Images loaded from: {}\nNumber of Images: {}".format(self.__path, len(self.images))


if __name__ == "__main__":

    path = "../data/cad_renders/GateRenders.jpg"
    path_load = "../data/cad_renders2_test"
    save_path = "../data/cad_renders2_dist"

    d = DistortImages()
    d.load(path_load, ".jpg")
    d.distort(save_path, blur=tuple(range(3,10,1)), compression=np.arange(5,15,1, dtype=np.uint8).tolist())


