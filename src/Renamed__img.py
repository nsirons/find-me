import os
from itertools import product
import re
from collections import defaultdict
from shutil import copyfile


class ImagePreProc:
    """
    This class contains total information about the images in the folder
    Could be used to rename all files in the specified order.
    """
    available_labels = ['img', 'R', 'alpha', 'beta', 'gamma', 'x', 'y', "blur", "gaussian_noise", "contrast",
                        "compression", "erosion", "dilation", "closing", "opening", "gradient"]

    def __init__(self):
        self.__path = None
        self.__save_path = None
        self.images_labels = None

    def load(self, path, ext=".jpg", background=False):
        if not os.path.exists(path):
            raise FileNotFoundError("Invalid path: {}".format(path))
        if self.__path is None:
            self.__path = path

        self.images_labels = defaultdict(dict)
        files = tuple(filter(lambda x: ext in x, os.listdir(self.__path)))
        labels = re.findall(r'[a-zA-Z]+', files[0][:-4])  # find all labels in file name, except extension

        if not background:
            for label in labels:
                if label not in self.available_labels:
                    raise KeyError("Invalid key, not present in available labels: {} not in {}".format(label, self.available_labels))

        for i, file_name in enumerate(files):
            digits = re.findall(r'[\d.]+', file_name[:-4])
            self.images_labels[i] = {"img":digits[0]}
            self.images_labels[i]['name'] = file_name
            for label, digit in zip(labels, digits):  # last label is image extension
                self.images_labels[i][label] = digit
            self.images_labels[i]['ext'] = ext

    def rename(self, path, order, save_path=None, ext=".jpg"):
        # order should have a structure {"R":(0, (0,1,2,3,4))), ....  "alpha":(5,(1,2,3,4,5))}
        # Check that path and keys are valid
        if self.__path is None:
            if not os.path.exists(path):
                raise FileNotFoundError("Invalid path: {}".format(path))
        else:
            raise FileExistsError("This class is already used by other path, please create new instance with this path")

        self.__save_path = save_path or path
        self.__path = path

        if not os.path.exists(self.__save_path):
            os.makedirs(self.__save_path)

        for key in order:
            if key not in self.available_labels:
                raise KeyError("Unknown key, following key is not in labels - {}\nkey: {}".format(self.available_labels, key))

        files = sorted(tuple(filter(lambda x: ext in x, os.listdir(self.__path))))

        # TODO: ugly
        params = len(order)*[None]
        keys = len(order)*[None]
        for key in order:
            params[order[key][0]] = order[key][1]
            keys[order[key][0]] = key
        combos = tuple(product(*params))

        if len(combos) != len(files):
            raise ArithmeticError("Number of files and combinations do not match: {} vs {}".format(len(files), len(combos)))

        # TODO: be sure not to overflow number of digits?
        template_name = "img{i:05d}_" + '_'.join([str(key)+"{{:4.5f}}".format() for key in keys]) + "{ext}"
        for i, combination in enumerate(combos):
            new_name = template_name.format(*combination, i=i, ext=ext)
            if self.__save_path == self.__path:
                os.rename(os.path.join(self.__path, files[i]), os.path.join(self.__save_path, new_name))
            else:
                copyfile(os.path.join(self.__path, files[i]), os.path.join(self.__save_path, new_name))

    def get_images(self):
        return self.images_labels

    def __len__(self):
        return len(self.images_labels)

    def __getitem__(self, item):
        return self.images_labels[item]

    def __str__(self):
        if self.images_labels is None:
            return "Not yet loaded"
        return "Image folder : {path}\nNumber of images: {len}\nImages format: {ext}\n" \
               "---------------------------------------\nParameters:\n" \
               "{params}".format(path=self.__path, len=len(self.images_labels), ext=self.images_labels[0]['ext'], params='\n'.join(self.images_labels[0].keys()))


if __name__ == "__main__":
    import numpy as np
    path = "/home/kit/projects/find-me/data/GateRenders 6/5-60"
    save_path = "/home/kit/projects/find-me/data/validation/5-60"
    rn = ImagePreProc()

    # rn.rename(path, {"R":(0,[3]), "alpha":(1, (0,)), "beta": (2, np.linspace(0,90,60))})
    rn.rename(path, {"R": (0, (5,)), "alpha": (1, (0,)), "beta": (2, np.linspace(60,0,15))}, save_path=save_path)
    # rn.load(path)
    # print(rn)
    # print(rn.get_images())