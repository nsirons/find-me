from src.benchmark import Benchmark
from src.template_matching import TemplateMatching
from src.Renamed__img import ImagePreProc
from src.image_distr import DistortImages
from src.find_corners import generate_label_file

import numpy as np

# Select images
paths = ["../data/cad_renders2", "../data/cad_renders3"]
path_corners = "../data/corners"

tm = TemplateMatching()
tm.load_corners(path_corners)

renamer = ImagePreProc()
distr = DistortImages()
bench = Benchmark(tm)

# Generate distorted images and load them new images
for path in paths:
    renamer.rename(path, {"R":(0,[3]), "alpha":(1, (0,)), "beta": (2, np.linspace(0,90,60))})
    distr.load(path, ".jpg")
    new_path = path + "_distr"
    distr.distort(new_path,
                  blur=(3, 5, 10),
                  compression=np.arange(5, 15, 1, dtype=np.uint8).tolist(),
                  dilation=(1, 3, 5),
                  erosion=(1, 3, 5))
    generate_label_file(new_path, ".jpg", True)
    bench.load(new_path)


data = bench.test()
print(data)