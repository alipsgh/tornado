"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import math
import operator
from collections import OrderedDict

from classifier.classifier import SuperClassifier
from dictionary.tornado_dictionary import TornadoDic


class DecisionStump(SuperClassifier):
    """The decision stump classifier. A decision stump is a decision tree only with the root node!"""

    LEARNER_NAME = TornadoDic.DECISION_STUMP
    LEARNER_TYPE = TornadoDic.TRAINABLE
    LEARNER_CATEGORY = TornadoDic.NOM_CLASSIFIER

    def __init__(self, labels, attributes):

        super().__init__(labels, attributes)

        self.CLASSES_DISTRIBUTIONS = OrderedDict()

        self.ATTRIBUTES_NAMES = []
        self.ATTRIBUTES_VALUES_DISTRIBUTIONS = OrderedDict()
        self.ATTRIBUTES_SCORES = OrderedDict()

        self.__STUMP = OrderedDict()

        self.__initialize_classes()
        self.__initialize_attributes()

    def __initialize_classes(self):
        for c in self.CLASSES:
            self.CLASSES_DISTRIBUTIONS[c] = 0

    def __initialize_attributes(self):
        for attr in self.ATTRIBUTES:
            self.ATTRIBUTES_NAMES.append(attr.NAME)
            self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr.NAME] = OrderedDict()
            for v in attr.POSSIBLE_VALUES:
                self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr.NAME][v] = OrderedDict()
                for c in self.CLASSES:
                    self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr.NAME][v][c] = 0

    def train(self, instance):
        self.NUMBER_OF_INSTANCES_OBSERVED += 1
        self.__set_class_dist(instance)
        self.__set_attr_val_dist(instance)
        self.__calculate_info_gain()
        self.__set_stump()

    def __set_class_dist(self, instance):
        y = instance[len(instance) - 1]
        self.CLASSES_DISTRIBUTIONS[y] += 1

    def __set_attr_val_dist(self, instance):
        x = instance[0:len(instance) - 1]
        y = instance[len(instance) - 1]
        for attr_index in range(0, len(x)):
            attr = self.ATTRIBUTES_NAMES[attr_index]
            value = x[attr_index]
            self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr][value][y] += 1

    def __calculate_info_gain(self):

        m = self.NUMBER_OF_INSTANCES_OBSERVED

        # CALCULATING EXPECTED INFORMATION
        expected_info_tr = 0
        for c, v in self.CLASSES_DISTRIBUTIONS.items():
            if v == 0:
                continue
            expected_info_tr += self.__calculate_entropy(v, m)

        # CALCULATING EXPECTED INFORMATION WITH CONSIDERING EACH ATTRIBUTE
        # THEN CALCULATING THEIR GAINS - OR SCORES
        for attr, values_and_distributions in self.ATTRIBUTES_VALUES_DISTRIBUTIONS.items():
            expected_info_attr = 0
            for value, class_distributions in values_and_distributions.items():
                sum_classes_dist = sum(class_distributions.values())
                expected_info_v = 0
                for class_key, class_dist in class_distributions.items():
                    if class_dist == 0:
                        continue
                    expected_info_v += self.__calculate_entropy(class_dist, sum_classes_dist)
                expected_info_attr += (sum_classes_dist / m) * expected_info_v
            self.ATTRIBUTES_SCORES[attr] = expected_info_tr - expected_info_attr

    @staticmethod
    def __calculate_entropy(x, y):
        entropy = (-1) * (x / y) * math.log2(x / y)
        return entropy

    def __set_stump(self):
        stump = max(self.ATTRIBUTES_SCORES.items(), key=operator.itemgetter(1))[0]
        self.__STUMP = {stump: self.ATTRIBUTES_VALUES_DISTRIBUTIONS[stump]}

    def test(self, instance):
        if self._IS_READY:
            x = instance[0:len(instance) - 1]
            y = instance[len(instance) - 1]
            attr = list(self.__STUMP.keys())[0]
            test_attr_value = x[self.ATTRIBUTES_NAMES.index(attr)]
            predicted_class = max(self.__STUMP[attr][test_attr_value].items(), key=operator.itemgetter(1))[0]
            self.update_confusion_matrix(y, predicted_class)
            return predicted_class
        else:
            print("Please train a Decision Stump classifier first!")
            exit()

    def reset(self):
        super()._reset_stats()
        self.CLASSES_DISTRIBUTIONS = OrderedDict()
        self.ATTRIBUTES_NAMES = []
        self.ATTRIBUTES_VALUES_DISTRIBUTIONS = OrderedDict()
        self.ATTRIBUTES_SCORES = OrderedDict()
        self.__STUMP = OrderedDict()
        self.__initialize_classes()
        self.__initialize_attributes()
