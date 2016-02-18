import numpy as np
import pandas as pd
from pyflann import FLANN
import part1 as pt1


def get_data(data_file):
    data = pd.read_csv(data_file, skiprows=2, header=None, sep="\t")
    names = data.iloc[:, 0].as_matrix()
    attrs = data.iloc[:, 2:].as_matrix()
    return attrs, names


def split_attrs(attrs):
    num_attrs = 73
    return attrs[:, :num_attrs], attrs[:, num_attrs:]


class ground_truth_classifier(object):

    def __init__(self, data_file):
        self.flann = FLANN()
        attributes, names = get_data(data_file)
        self.flann.build_index(attributes, algorithm="autotuned")
        self.names = names

    def predict(self, attrs):
        attrs1, attrs2 = split_attrs(attrs)
        idx1, _ = self.flann.nn_index(attrs1)
        names1 = self.names[idx1]
        idx2, _ = self.flann.nn_index(attrs2)
        names2 = self.names[idx2]
        return (names1 == names2)


def test_ground_truth(names):
    model = ground_truth_classifier('./data/pubfig_attributes.txt')
    for name in names:
        (x, y) = pt1.read_data(names)
        y_ = ground_truth_classifier()
        print("the data from %s has accuracy of %f"%(name, np.sum(y==y_)/y.size))
