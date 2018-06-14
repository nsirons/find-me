from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import cv2

# TODO: NOT USED ANYWERE (FOR NOW)
class LabelImage:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.images = {}

        self.load()

    def load(self):
        mat_path = tuple(filter(lambda f_name: 'mat' in f_name, os.listdir(self.dir_path)))
        if len(mat_path) != 1:
            raise NameError("Invalid mat file in {}, there are {} mat files, must be 1".format(self.dir_path, len(mat_path)))

        label_data = loadmat("{}/{}".format(self.dir_path, mat_path[0]))

        for i, img_path in enumerate(sorted(filter(lambda f_name: 'jpg' in f_name, os.listdir(self.dir_path)))):
            img_int = int(img_path[4:-4])
            self.images[img_int] = {"image": cv2.imread("{}/{}".format(self.dir_path, img_path)),
                                    "label": label_data['GT_gate'][i]}

    def __str__(self):
        return str(len(self.images.keys()))

    def get_dataset(self):
        return self.images


if __name__ == "__main__":
    # dir_path = "data/cyberzoo_distance/"
    dir_path = "data/pic_cyberzoo"
    li = LabelImage(dir_path)

    h = li.images[0]['image'].shape[1]
    for img in range(10):
        img = li.images[img]
        for i in range(4):
            img['image'] = cv2.circle(img['image'], (int(img['label'][1+i]), h-int(img['label'][5+i]))[::-1], 1, (255,0,0),4)

        plt.imshow(img['image'], origin='lowering')
        # plt.show()
    # f = loadmat("data/cyberzoo_distance/4_16_cyberzoo_dis.mat")
    # print(f['GT_gate'].shape)
    # print(f.keys())
    # print(f['dir_name'])