import os
from PIL import Image
import numpy as np
import cv2

path_download = "downloads/all"
path = "../data/cnn_inputs"
save_path = "../data/cnn_inputs_compr"
path_target = "../data/cnn_targets"

def compress(path, save_path):
    quality_val = 5


    print("Compressing ")
    for file_name in os.listdir(path):
        if '.jpg' in file_name:
            im = Image.open(os.path.join(path, file_name))
            im.save(os.path.join(save_path, file_name), format='JPEG', subsampling=0, quality=quality_val)
    print("Compression done!")

def bg_to_tf_inputs_targers(path, path_tergets):
    print("Parsing background images to cnn")
    empty_img = 255*np.ones((288, 512))
    for file_name in os.listdir(path):
        if '.jpg' in file_name or '.png' in file_name:
            try:
                cv2.imwrite(os.path.join(path_tergets, file_name), empty_img)
            except:
                print(file_name)
    print("Parsing done")

def check_img(p1, p2):
    pp2 = os.listdir(p2)
    for id, i in enumerate(os.listdir(p1)):
        if i in pp2:
            try:
                img = cv2.imread(os.path.join(p1,i)).shape
                cv2.resize(img,(512,288))
            except:
                # img.resize((512, 288))
                print(i)
        else:
            os.remove(os.path.join(p1, i))
            print(i)
    print("DONE")
# cv2.imwrite()
# bg_to_tf_inputs_targers(path_download, path_target)
# cv2.imwrite("downloads/BIT.png", 255*np.ones((288,512)) )
check_img(save_path, path_target)
# cv2.resize()