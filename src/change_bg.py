import cv2
import numpy as np
import os
from shutil import copyfile
import random
from src.image_distr import DistortImages


def separate(path, save_path):
    """
    This function separates object images from color images, in order to generate dataset for CNN.
    :param path: str
    :return: None
    """

    if not os.path.exists(path):
        raise FileNotFoundError("Invalid path: {}".format(path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path, "targets")):
        os.makedirs(os.path.join(save_path, "targets"))
        os.makedirs(os.path.join(save_path, 'inputs'))

    for file_name in os.listdir(path):
        if "_Object" in file_name:
            f_name = os.path.join(save_path, "targets", file_name[:-len("_Object.jpg")] + file_name[-len(".jpg"):])
            copyfile(os.path.join(path, file_name), f_name)
            img = cv2.imread(f_name)
            flip = cv2.flip(img, flipCode=0)
            rever = cv2.rotate(img, rotateCode=cv2.ROTATE_180)
            flip_name = os.path.join(save_path, "targets", file_name[:-len("_Object.jpg")] + "_flip" + file_name[-len(".jpg"):])
            rever_name = os.path.join(save_path, "targets", file_name[:-len("_Object.jpg")] + "_revers" + file_name[-len(".jpg"):])
            cv2.imwrite(flip_name, flip)
            cv2.imwrite(rever_name, rever)
        else:
            f_name = os.path.join(save_path, "inputs", file_name)
            copyfile(os.path.join(path, file_name), f_name)
            img = cv2.imread(f_name)
            flip = cv2.flip(img, flipCode=0)
            rever = cv2.rotate(img, rotateCode=cv2.ROTATE_180)
            flip_name = os.path.join(save_path, "inputs", file_name[:-len(".jpg")] + "_flip" + file_name[-len(".jpg"):])
            rever_name = os.path.join(save_path, "inputs", file_name[:-len(".jpg")] + "_revers" + file_name[-len(".jpg"):])
            cv2.imwrite(flip_name, flip)
            cv2.imwrite(rever_name, rever)


def change_bg(load_bg, core_path, number_of_new):
    bg_lst = []
    for bg in filter(lambda x: 'txt' not in x, os.listdir(load_bg)):
        bg_img = cv2.imread(os.path.join(load_bg, bg))
        if bg_img is not None:
            bg_img = cv2.resize(bg_img, (512, 288))

            # print(bg_img)

            assert bg_img.shape == (288, 512, 3)
            # assert False
            bg_lst.append(bg_img)

    for i, image_name in enumerate(random.choices(os.listdir(os.path.join(core_path, 'inputs')), k=number_of_new)):
        mask = cv2.imread(os.path.join(core_path, "targets", image_name), 0)
        ret, mask = cv2.threshold(mask, 20,255, cv2.THRESH_BINARY)
        img = cv2.imread(os.path.join(core_path, "inputs", image_name))
        bg = random.choice(bg_lst)
        mask_inv = cv2.bitwise_not(mask)
        gate = cv2.bitwise_and(img, img, mask=mask)
        new_bg = cv2.bitwise_and(bg, bg, mask=mask_inv)
        new_gate_with_bg = new_bg + gate
        new_f_name = image_name[:-4] + "random{:05d}".format(i) + image_name[-4:]
        cv2.imwrite(os.path.join(core_path, "inputs", new_f_name), new_gate_with_bg)
        cv2.imwrite(os.path.join(core_path, "targets", new_f_name), mask)


def jpeg_all_inputs(path):
    d = DistortImages()
    d.load(os.path.join(path, "inputs"), ".jpg")
    d.distort(os.path.join(path, "inputs_mod"), compression=(10,), blur=((10,),(3,3)), gaussian_noise=((10,),))
    for file_name in os.listdir(os.path.join(path, 'targets')):
        new_name_compr = file_name[:-4] + "_compression{:03d}".format(10) + file_name[-4:]
        new_name_blur = file_name[:-4] + "_blur{:03d}".format(10) + file_name[-4:]
        new_name_noise = file_name[:-4] + "_gaussian_noise{:03d}".format(10) + file_name[-4:]
        copyfile(os.path.join(path, 'targets', file_name), os.path.join(path, 'targets', new_name_compr))
        copyfile(os.path.join(path, 'targets', file_name), os.path.join(path, 'targets', new_name_blur))
        copyfile(os.path.join(path, 'targets', file_name), os.path.join(path, 'targets', new_name_noise))
        # os.rename(os.path.join(path, 'targets', file_name), os.path.join(path, 'targets', new_name))
        os.remove(os.path.join(path, 'targets', file_name))


def check(path):
    inputs = os.listdir(os.path.join(path, 'inputs_mod'))
    for file_name in os.listdir(os.path.join(path, 'targets')):
        if file_name not in inputs:
            print(file_name)
            assert False
    print("CHECK - OK")


def add_fake(from_path, save_path):
    empty = np.zeros((288, 512, 3))
    for img_name in os.listdir(from_path):
        try:
            img = cv2.imread(os.path.join(from_path, img_name))
            # bg_img = cv2.resize(bg_img, (512, 288))
            img = cv2.resize(img, (512, 288))
            cv2.imwrite(os.path.join(save_path, "inputs_mod", img_name), img)
            cv2.imwrite(os.path.join(save_path, "targets", img_name), empty)
        except cv2.error:
            print("REMOVED: ", img_name)
            os.remove(os.path.join(from_path, img_name))


if __name__ == "__main__":
    path_to_bg = "../data/all"
    paths = "../data/20June2/"
    to_path = "../data/cnn_data7"
    # all_paths = os.listdir(paths)
    # for path in all_paths:
    #     # print(os.listdir(os.path.join(paths, path)))
    #     folder_in_folder = os.listdir(os.path.join(paths, path))[0]
    #     separate(os.path.join(paths, path, folder_in_folder), to_path)
    # change_bg(path_to_bg, to_path, 3000)
    # jpeg_all_inputs(to_path)
    # add_fake('../data/all', to_path)
    check(to_path)
