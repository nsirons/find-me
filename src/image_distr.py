from PIL import Image
import cv2
import os
import numpy as np
from src.GeneralImage import GeneralImage
import matplotlib.pyplot as plt


def gaussian_noise(img, mu, sigma):
    return np.clip(img + np.random.normal(mu, sigma, img.shape).astype(np.uint8), 0, 255)

path = "../data/cad_renders"
save_path = "../data/cad_renders_dist"

# all the coefficients are taken from: https://arxiv.org/pdf/1604.04004.pdf
img_name = save_path+'/img_{:d}_{:s}_{:d}.jpg'

if not os.path.exists(save_path):
    print("New folder created: {}".format(save_path))
    os.makedirs(save_path)

print("Deleting images in: {}".format(save_path))
#  clean folder
for file_delete in os.listdir(save_path):
    file_path = os.path.join(save_path, file_delete)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

print("Generating new images")
for i, file_name in enumerate(sorted(os.listdir(path))):
    img = GeneralImage(path + '/' + file_name)

    for quality_val in range(10, 100, 10):
        im = Image.open(path + '/' + file_name)
        im.save(img_name.format(i,"jpeg", quality_val), format='JPEG', subsampling=0,quality=quality_val)

    for contrast in range(1, 10):
        cv2.imwrite(img_name.format(i, "contrast", contrast * 10), contrast / 10 * img.bgr())

    for sigma in range(1, 11):
        cv2.imwrite(img_name.format(i, "blur", sigma),
                    cv2.GaussianBlur(img.bgr(), (11, 11), sigmaX=sigma, sigmaY=sigma))  # TODO: change size of kernel?

    for sigma in range(10, 100, 10):
        cv2.imwrite(img_name.format(i, "noise", sigma), gaussian_noise(img.rgb(), 0, sigma))
