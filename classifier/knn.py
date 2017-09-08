"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import math
import operator

from classifier.classifier import SuperClassifier
from dictionary.tornado_dictionary import *


def calculate_euclidean_distance(instance_1, instance_2):
    summation = 0
    for i in range(0, len(instance_1)):
        summation += math.pow((float(instance_1[i]) - float(instance_2[i])), 2)
    distance = math.sqrt(summation)
    return distance


def calculate_hamming_distance(instance_1, instance_2):
    n = len(instance_1)
    summation = 0
    for i in range(0, len(instance_1)):
        if instance_1[i] != instance_2[i]:
            summation += 1
    distance = summation / n
    return distance


class KNN(SuperClassifier):
    """This is the implementation of the K Nearest Neighbor algorithm. This classifier keeps the recent
    instances of a data stream within a window. For a new instance, its nearest neighbors are located in the window.
    Then, using the majority voting approach, the class of the new instance is decided."""

    LEARNER_NAME = TornadoDic.K_NN
    LEARNER_TYPE = TornadoDic.LOADABLE
    LEARNER_CATEGORY = TornadoDic.NUM_CLASSIFIER

    def __init__(self, labels, attributes, k=5, window_size=100):

        super().__init__(labels, attributes)

        self.INSTANCES = []
        self.K = k
        self.LEARNER_NAME = str(self.K) + " NEAREST NEIGHBORS"
        self.__WINDOW_SIZE = window_size

    def load(self, instance):
        if len(self.INSTANCES) > self.__WINDOW_SIZE:
            self.INSTANCES.pop(0)
        self.INSTANCES.append(instance)

    def test(self, ts_instance):
        if self._IS_READY:
            distances = []
            x_test = ts_instance[0:len(ts_instance) - 1]
            y = ts_instance[len(ts_instance) - 1]
            for instance in self.INSTANCES:
                x = instance[0:len(instance) - 1]
                distances.append([instance, ts_instance, calculate_euclidean_distance(x, x_test)])
            knn = self.__find_k_nearest_neighbours(distances)
            predicted_class = self.__predict(knn)
            self.update_confusion_matrix(y, predicted_class)
            return predicted_class
        else:
            print("Please load KNN classifier with some instances first!")
            exit()

    def __find_k_nearest_neighbours(self, distances):
        unsorted_distance_dic = {}
        for i in range(0, len(distances)):
            unsorted_distance_dic[i] = distances[i][2]
        sorted_distance_list = sorted(unsorted_distance_dic.items(), key=operator.itemgetter(1))
        knn = {}
        num_total_nodes = len(distances)
        if num_total_nodes < self.K:
            k = num_total_nodes
        else:
            k = self.K
        for i in range(0, k):
            knn[i] = distances[sorted_distance_list[i][0]][0]
        return knn

    @staticmethod
    def __predict(knn):
        knn_class_dist = {}
        for k, v in knn.items():
            if knn_class_dist.__contains__(v[len(v) - 1]) is True:
                knn_class_dist[v[len(v) - 1]] += 1
            else:
                knn_class_dist[v[len(v) - 1]] = 1
        prediction = max(knn_class_dist.items(), key=operator.itemgetter(1))[0]
        return prediction

    def reset(self):
        super()._reset_stats()
        self.INSTANCES = []
