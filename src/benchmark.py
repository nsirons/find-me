import matplotlib.pyplot as plt
import os
import csv
import numpy as np

# TODO: Need to be tested
# TODO: Strongly assumed to be sorted?


class Benchmark:
    def __init__(self, cls):
        self.gate_detection_class = cls  # Must be a function that return coordinates of the image
        self.validation_data_path = None
        self.validation_data = None
        self.current_result = None
        self.stats = None

    def load_data(self, path):
        self.validation_data = []
        self.validation_data_path = path
        label_file = tuple(filter(lambda x: 'txt' in x, os.listdir(path)))
        if len(label_file) != 1:
            raise FileNotFoundError("Invalid path: {}".format(path))
        with open("{}/{}".format(path, label_file[0]), mode='r') as csvfile:
            reader = csv.DictReader(csvfile, dialect="excel-tab")
            for row in reader:
                self.validation_data.append(row)
        print(self.validation_data[0])

    def test(self):
        self.current_result = {}
        self.stats = {}
        counter = 0
        for i, file in enumerate(filter(lambda x: 'jpg' in x, sorted(os.listdir(self.validation_data_path)))):
            # if file[-3:] in ['png', 'jpg']:
            corners = self.gate_detection_class.find_gate(os.path.join(self.validation_data_path, file))  # every class should have a function
            gate_det, p0,p1,p2,p3 = corners
            self.current_result[i] = {}
            if int(self.validation_data[i]['gate']) == gate_det:
                counter += 1
                self.current_result[i]["Gate detection"] = 1
                self.current_result[i]["x0"] = abs(p0[0] - float(self.validation_data[i]["x0"]))
                self.current_result[i]["x1"] = abs(p1[0] - float(self.validation_data[i]["x1"]))
                self.current_result[i]["x2"] = abs(p2[0] - float(self.validation_data[i]["x2"]))
                self.current_result[i]["x3"] = abs(p3[0] - float(self.validation_data[i]["x3"]))
                self.current_result[i]["y0"] = abs(p0[1] - float(self.validation_data[i]["y0"]))
                self.current_result[i]["y1"] = abs(p1[1] - float(self.validation_data[i]["y1"]))
                self.current_result[i]["y2"] = abs(p2[1] - float(self.validation_data[i]["y2"]))
                self.current_result[i]["y3"] = abs(p3[1] - float(self.validation_data[i]["y3"]))
            else:
                self.current_result[i]["Gate detection"] = 0

        # self.current_result["Gate detection"] /= len(self.validation_data)
        # for i in range(4):
        #     self.current_result["p{}".format(i)] /= len(self.validation_data)
        self.stats["gate"] = counter / i
        mean_error = np.array([(i["x0"], i["x1"], i["x2"], i["x3"],
                                i["y0"], i["y1"], i["y2"], i["y3"]) for i in self.current_result.values() if i["Gate detection"] ==1])
        self.stats["error"] = {}
        self.stats["error"]["mean"] = np.mean(mean_error, axis=1)
        self.stats["error"]["max"] = np.max(mean_error, axis=1)
        self.stats["error"]["min"] = np.min(mean_error, axis=1)
        print(mean_error)

    def get_stats(self):
        return self.current_result

    def __str__(self):
        if self.stats is None:
            return "Nothing has been loaded"
        return "Benchmarking: {}\n" \
               "Validation dataset path: {}\n" \
               "Gate detection: {:3f}\n" \
               "Corner Accuracy Filtered:\n" \
               "p  \tmin  \tmax  \tave  \n" \
               "x0: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
               "x1: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
               "x2: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
               "x3: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
               "y0: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
               "y1: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
               "y2: {:3.3f}\t{:3.3f}\t{:3.3f}\n" \
               "y3: {:3.3f}\t{:3.3f}\t{:3.3f}\n". \
                    format(self.gate_detection_class.__class__.__name__,
                       self.validation_data_path, self.stats["gate"]*100,
                           self.stats["error"]["min"][0],self.stats["error"]["max"][0],self.stats["error"]["mean"][0],
                           self.stats["error"]["min"][1], self.stats["error"]["max"][1], self.stats["error"]["mean"][1],
                           self.stats["error"]["min"][2], self.stats["error"]["max"][2], self.stats["error"]["mean"][2],
                           self.stats["error"]["min"][3], self.stats["error"]["max"][3], self.stats["error"]["mean"][3],
                           self.stats["error"]["min"][4], self.stats["error"]["max"][4], self.stats["error"]["mean"][4],
                           self.stats["error"]["min"][5], self.stats["error"]["max"][5], self.stats["error"]["mean"][5],
                           self.stats["error"]["min"][6], self.stats["error"]["max"][6], self.stats["error"]["mean"][6],
                           self.stats["error"]["min"][7], self.stats["error"]["max"][7], self.stats["error"]["mean"][7])

if __name__ == "__main__":
    path = "../data/cad_renders2"
    a = Benchmark(1)
    a.load_data(path)

    print(a)
