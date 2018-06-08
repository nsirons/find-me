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

    def load_data(self, path):
        self.validation_data = []
        self.validation_data_path = path
        label_file = tuple(filter(lambda x: 'txt' in x, os.listdir(path)))
        if len(label_file) != 1:
            raise FileNotFoundError("Invalid path: {}".format(path))
        with open("{}/{}".format(path, label_file[0]), mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.validation_data.append(row)

    def test(self):
        self.current_result = []
        for file in os.listdir(self.validation_data_path):
            if file[-3:] in ['png', 'jpg']:
                corners = self.gate_detection_class.find_gate(file)  # every class should have a function
                # TODO: compare corners

    def get_stats(self):
        return self.current_result

    def __str__(self):
        return "Benchmarking: {}\n" \
               "Validation dataset path: {}\n" \
               "Gate detection: {}\n" \
               "Corner Accuracy Filtered: {}\n".\
                format(self.gate_detection_class.__class__.__name__,
                       self.validation_data_path, np.sum(self.current_result[:, 0]), 1)

if __name__ == "__main__":
    a = Benchmark(1)
    print(a.get_stats().__class__.__name__)