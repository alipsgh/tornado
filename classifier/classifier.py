"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import time
from collections import OrderedDict

from dictionary.tornado_dictionary import TornadoDic
from evaluators.classifier_evaluator import PredictionEvaluator


class SuperClassifier:
    """A classifier, e.g. Naive Bayes, inherits this super class!"""

    def __init__(self, labels, attributes):
        self.CLASSES = labels
        self.ATTRIBUTES = attributes
        self.__CONFUSION_MATRIX = OrderedDict()
        self.__GLOBAL_CONFUSION_MATRIX = OrderedDict()

        self.NUMBER_OF_INSTANCES_OBSERVED = 0

        self._TRAINING_TIME = 0
        self._TESTING_TIME = 0
        self._TOTAL_TRAINING_TIME = 0
        self._TOTAL_TESTING_TIME = 0

        # THE _ACTIVE ATTRIBUTE IS USED TO SHOW WHETHER A CLASSIFIER IS SUSPENDED OR NOT
        self._ACTIVE = True
        self._IS_READY = False

        self.__init_confusion_matrix()
        self.__init_global_confusion_matrix()

    def __init_confusion_matrix(self):
        # self.__CONFUSION_MATRIX =
        for real_class in self.CLASSES:
            self.__CONFUSION_MATRIX[real_class] = OrderedDict()
            for predicted_class in self.CLASSES:
                self.__CONFUSION_MATRIX[real_class][predicted_class] = 0

    def __init_global_confusion_matrix(self):
        # self.__GLOBAL_CONFUSION_MATRIX =
        for real_class in self.CLASSES:
            self.__GLOBAL_CONFUSION_MATRIX[real_class] = OrderedDict()
            for predicted_class in self.CLASSES:
                self.__GLOBAL_CONFUSION_MATRIX[real_class][predicted_class] = 0

    def update_confusion_matrix(self, real_class, predicted_class):
        self.__CONFUSION_MATRIX[real_class][predicted_class] += 1
        self.__GLOBAL_CONFUSION_MATRIX[real_class][predicted_class] += 1

    def get_confusion_matrix(self):
        return self.__CONFUSION_MATRIX

    def print_confusion_matrix(self):
        for real_class in self.CLASSES:
            for predicted_class in self.CLASSES:
                print(self.__CONFUSION_MATRIX[real_class][predicted_class], end="\t")
            print()

    def get_global_confusion_matrix(self):
        return self.__GLOBAL_CONFUSION_MATRIX

    def reset_confusion_matrix(self):
        self.__init_confusion_matrix()

    def is_ready(self):
        return self._IS_READY

    def set_ready(self):
        self._IS_READY = True

    def is_active(self):
        return self._ACTIVE

    def deactivate(self):
        self._ACTIVE = False

    def activate(self):
        self._ACTIVE = True

    def get_training_time(self):
        return self._TRAINING_TIME

    def get_testing_time(self):
        return self._TESTING_TIME

    def get_running_time(self):
        return self._TRAINING_TIME + self._TESTING_TIME

    def get_total_running_time(self):
        return self._TOTAL_TRAINING_TIME + self._TOTAL_TESTING_TIME

    def get_error(self):
        return PredictionEvaluator.calculate(TornadoDic.ERROR_RATE, self.get_confusion_matrix())

    def _reset_stats(self):
        # HERE I NEED TO MAKE SOME MODIFICATIONS
        # FOR CONSIDERING CONCEPT DRIFTS.
        self.NUMBER_OF_INSTANCES_OBSERVED = 0
        self._TRAINING_TIME = 0
        self._TESTING_TIME = 0
        self._IS_READY = False
        self._ACTIVE = True

        self.reset_confusion_matrix()

    def do_training(self, record):
        t1 = time.perf_counter()
        self.train(record)
        t2 = time.perf_counter()
        delta = (t2 - t1) * 1000  # in milliseconds
        self._TRAINING_TIME += delta
        self._TOTAL_TRAINING_TIME += delta

    def train(self, record):
        pass

    def do_loading(self, record):
        t1 = time.time()
        self.load(record)
        t2 = time.time()
        delta = (t2 - t1) * 1000  # in milliseconds
        self._TRAINING_TIME += delta
        self._TOTAL_TRAINING_TIME += delta

    def load(self, record):
        pass

    def do_testing(self, record):
        t1 = time.time()
        pr = self.test(record)
        t2 = time.time()
        delta = (t2 - t1) * 1000  # in milliseconds
        self._TESTING_TIME += delta
        self._TOTAL_TESTING_TIME += delta
        return pr

    def test(self, record):
        pass
