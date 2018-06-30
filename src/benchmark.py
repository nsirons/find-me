import csv
import os
from collections import defaultdict
from itertools import product
import progressbar
import numpy as np
import time
import matplotlib.pyplot as plt

# TODO: Need to be tested
# TODO: Strongly assumed to be sorted?


class Benchmark:
    def __init__(self, cls):
        self.gate_detection_class = cls  # Must be a function that return coordinates of the image
        self.validation_data_path = None
        self.validation_data = None
        self.current_result = None
        self.stats = None
        self.roc_stats = None

    def load_data(self, path):
        self.validation_data = []
        self.validation_data_path = []
        self.load(path)

    def load(self, path):
        if self.validation_data is None:
            self.validation_data = []
        if self.validation_data_path is None:
            self.validation_data_path = []

        store_stash = {}
        self.validation_data_path.append(path)
        label_file = tuple(filter(lambda x: 'txt' in x, os.listdir(path)))
        if len(label_file) != 1:
            raise FileNotFoundError("Invalid path (not text file is there): {}".format(path))
        with open("{}/{}".format(path, label_file[0]), mode='r') as csvfile:
            reader = csv.DictReader(csvfile, dialect="excel-tab")
            for row in reader:
                store_stash[row["Name"]] = row
        self.validation_data.append(store_stash)

    def test_batch(self, dataset_id=None, *args, **kwargs):
        data = []
        for combination in product(*args):
            data.append(self.test(*combination, **kwargs))
        return data

    def test(self, dataset_id=None, *args, **kwargs):
        # if dataset_id is None - chech all data, else id
        if self.validation_data_path is None:
            raise FileNotFoundError("Dataset is not loaded")
        print("Running: ", args)

        self.current_result = defaultdict(dict)
        self.stats = defaultdict(dict)

        if dataset_id is None:
            for new_dataset_id, validation_dataset_path in enumerate(self.validation_data_path):
                self.calc_stats(new_dataset_id, validation_dataset_path, *args)
        else:
            self.calc_stats(dataset_id, self.validation_data_path[dataset_id], *args)

        print("Benchmark Finished !")

        for dataset_id in self.current_result:
            gate_detection = tuple(map(lambda x : x["GD"], self.current_result[dataset_id].values()))
            for gd_type in ["TP", "TN", "FP", "FN"]:
                self.stats[dataset_id][gd_type] = gate_detection.count(gd_type)

            TP_data =tuple(filter(lambda x: x["GD"] == "TP", self.current_result[dataset_id].values()))

            self.stats[dataset_id]["error"] = {}
            if TP_data:
                mean_error = np.array([(i["x0"], i["x1"], i["x2"], i["x3"],
                                        i["y0"], i["y1"], i["y2"], i["y3"]) for i in TP_data])

                self.stats[dataset_id]["error"]["mean"] = np.mean(mean_error, axis=0)
                self.stats[dataset_id]["error"]["max"] = np.max(mean_error, axis=0)
                self.stats[dataset_id]["error"]["min"] = np.min(mean_error, axis=0)
            else:
                self.stats[dataset_id]["error"]["mean"] = 0
                self.stats[dataset_id]["error"]["max"] = 0
                self.stats[dataset_id]["error"]["min"] = 0

            # TODO: add total statistics
        self.stats["total"]["ACC"] = sum([self.stats[key]["TP"]+self.stats[key]["TN"] for key in self.current_result]) / \
                                     sum([len(self.current_result[key]) for key in self.current_result])

        return self.stats

    def calc_stats(self, dataset_id, path, *args):
        with progressbar.ProgressBar(redirect_stdout=True, max_value=len(self.validation_data[dataset_id])) as bar:
            print("Benchmarking: {}".format(path), *args)
            for i, file in enumerate(self.validation_data[dataset_id]):
                corners = self.gate_detection_class.find_gate(os.path.join(path, file), *args)  # every class should have a function
                gate_det, p0, p1, p2, p3 = corners
                self.current_result[dataset_id][file] = defaultdict(dict)
                # print(self.validation_data[dataset_id].keys())
                if bool(int(self.validation_data[dataset_id][file]['gate'])) == gate_det:
                    if bool(int(self.validation_data[dataset_id][file]['gate'])):
                        # Gate is present and detected
                        self.current_result[dataset_id][file]["x0"] = abs(p0[0] - float(self.validation_data[dataset_id][file]["x0"]))
                        self.current_result[dataset_id][file]["x1"] = abs(p1[0] - float(self.validation_data[dataset_id][file]["x1"]))
                        self.current_result[dataset_id][file]["x2"] = abs(p2[0] - float(self.validation_data[dataset_id][file]["x2"]))
                        self.current_result[dataset_id][file]["x3"] = abs(p3[0] - float(self.validation_data[dataset_id][file]["x3"]))
                        self.current_result[dataset_id][file]["y0"] = abs(p0[1] - float(self.validation_data[dataset_id][file]["y0"]))
                        self.current_result[dataset_id][file]["y1"] = abs(p1[1] - float(self.validation_data[dataset_id][file]["y1"]))
                        self.current_result[dataset_id][file]["y2"] = abs(p2[1] - float(self.validation_data[dataset_id][file]["y2"]))
                        self.current_result[dataset_id][file]["y3"] = abs(p3[1] - float(self.validation_data[dataset_id][file]["y3"]))
                        x_list = tuple(filter(lambda x: x is not None, [i if float(self.validation_data[dataset_id][file]["x" + str(i)]) != 0 else None for i in range(4)]))
                        y_list = tuple(filter(lambda x: x is not None, [i if float(self.validation_data[dataset_id][file]["y" + str(i)]) != 0 else None for i in range(4)]))
                        xs = [self.current_result[dataset_id][file]["x"+str(i)] for i in x_list]
                        ys = [self.current_result[dataset_id][file]["y"+str(i)] for i in y_list]
                        if np.max(xs) >50 or np.max(ys) > 50:
                            self.current_result[dataset_id][file]["GD"] = "FN"
                        else:
                            self.current_result[dataset_id][file]["GD"] = "TP"

                    else:
                        # Gate is absent and not detected
                        self.current_result[dataset_id][file]["GD"] = "TN"
                else:
                    if bool(int(self.validation_data[dataset_id][file]['gate'])):
                        # Gate is present and not detected
                        self.current_result[dataset_id][file]["GD"] = "FP"
                    else:
                        # Gate is absent and is detected
                        self.current_result[dataset_id][file]["GD"] = "FN"
                time.sleep(0.01)
                bar.update(i)

    def get_stats(self):
        return self.current_result

    def false_positive_curve(self):
        TPR = [0]
        FPR = [0]
        TPs = []
        FNs = []
        CPs = []
        for key in self.current_result:
            TPs.append(self.stats[key]["TP"])
            FNs.append(self.stats[key]["FN"])
            CPs.append(sum(map(lambda x: int(x["gate"]), self.validation_data[key].values())))

            TPR.append(TPs[-1] / CPs[-1])
            FPR.append(FNs[-1] / CPs[-1])
            plt.plot(FPR[-1], TPR[-1], marker='*', color="blue", label=key)
            print(FPR, TPR)
        plt.title("Line of success")
        plt.plot([0, 1], [0, 1], label="Random guess", linestyle='--', color='red')
        plt.ylim((0, 1))
        plt.xlim((0, 1))
        plt.xlabel("FPR (False Positive Rate)")
        plt.ylabel("TPR (True Positive Rate)")
        plt.legend()
        plt.show()

    def __str__(self):
        if self.stats is None:
            return "Nothing has been loaded"
        return "Benchmarking: {}\n" \
               "Validation dataset path: {}\n" \
               "Gate detection: {:3f}\n" \
               "Corner Accuracy Filtered:\n" \
               "p  \tmin  \tmax  \tave  \n". \
                    format(self.gate_detection_class.__class__.__name__,
                           self.validation_data_path, self.stats["total"]["ACC"]*100,)
                           # self.stats["total"]["error"]["min"][0],self.stats["total"]["error"]["max"][0],self.stats["total"]["error"]["mean"][0],
                           # self.stats["total"]["error"]["min"][1], self.stats["total"]["error"]["max"][1], self.stats["total"]["error"]["mean"][1],
                           # self.stats["total"]["error"]["min"][2], self.stats["total"]["error"]["max"][2], self.stats["total"]["error"]["mean"][2],
                           # self.stats["total"]["error"]["min"][3], self.stats["total"]["error"]["max"][3], self.stats["total"]["error"]["mean"][3],
                           # self.stats["total"]["error"]["min"][4], self.stats["total"]["error"]["max"][4], self.stats["total"]["error"]["mean"][4],
                           # self.stats["total"]["error"]["min"][5], self.stats["total"]["error"]["max"][5], self.stats["total"]["error"]["mean"][5],
                           # self.stats["total"]["error"]["min"][6], self.stats["total"]["error"]["max"][6], self.stats["total"]["error"]["mean"][6],
                           # self.stats["total"]["error"]["min"][7], self.stats["total"]["error"]["max"][7], self.stats["total"]["error"]["mean"][7])

                            # "x0: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
                            # "x1: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
                            # "x2: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
                            # "x3: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
                            # "y0: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
                            # "y1: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
                            # "y2: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
                            # "y3: {:3.3f}\t{:3.3f}\t{:3.3f}\n". \


if __name__ == "__main__":
    # harris 26.66
    # template 33.33
    path2 = "../data/cad_renders2_dist"
    path_bg = "../data/backgrounds"
    path3 = "../data/cad_renders"
    path5 = "../data/cad_renders5_dist"
    # path6 = "/home/kit/projects/find-me/data/20June2/GateRenders360Rotation/GateRenders360Rotation_animation"
    path_valid_35 = "../data/validation/3-5"
    path_valid_360 = "../data/validation/3-60"
    path_valid_53 = "../data/validation/5-3"
    path_valid_560 = "../data/validation/5-60"
    from src.template_matching import TemplateMatching
    from src.harris_detector import Harris_detector

    harrs = Harris_detector()

    path_corners = "../data/corners"
    tm = TemplateMatching()
    tm.load_corners(path_corners)

    b = Benchmark(tm)
    b.load(path_valid_35)
    b.load(path_valid_360)
    b.load(path_valid_53)
    b.load(path_valid_560)

    # b.load(path_bg)
    # b.load(path2)
    # b.load(path3)
    # b.load(path)
    a = b.test(None, "cv2.TM_CCORR_NORMED", 0.95)
    print(b)
    print(a)
    # b.false_positive_curve()
