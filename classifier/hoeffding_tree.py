"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
** The Hoeffding Tree Implementation **
Paper: Domingos, Pedro, and Geoff Hulten. "Mining high-speed data streams."
Published in: Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2000.
URL: https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf
"""

import gc
import math
import operator
from collections import OrderedDict

from classifier.classifier import SuperClassifier
from dictionary.tornado_dictionary import TornadoDic


def calculate_hoeffding_bound(r, delta, n):
    epsilon = math.sqrt((math.pow(r, 2) * math.log((1 / delta), math.e)) / (2 * n))
    return epsilon


def calculate_entropy(x, y):
    entropy = (-1) * (x / y) * math.log2(x / y)
    return entropy


def calculate_info_gain(node):
    """This function calculate the information gain of attributes given a node."""

    # CALCULATING EXPECTED INFORMATION OF WHOLE TRAINING DATA
    expected_info_tr = 0
    for c, v in node.CLASSES_DISTRIBUTIONS.items():
        if v == 0:
            continue
        expected_info_tr += calculate_entropy(v, node.NUMBER_OF_EXAMPLES_SEEN)

    # CALCULATING EXPECTED INFORMATION WITH CONSIDERING EACH ATTRIBUTE
    # THEN CALCULATING THEIR GAINS - OR SCORES
    for attr, values_and_distributions in node.CANDIDATE_ATTRIBUTES_VALUES_DISTRIBUTIONS.items():
        expected_info_attr = 0
        for value, class_distributions in values_and_distributions.items():
            sum_classes_dist = sum(class_distributions.values())
            expected_info_v = 0
            for class_key, class_dist in class_distributions.items():
                if class_dist == 0:
                    continue
                expected_info_v += calculate_entropy(class_dist, sum_classes_dist)
            expected_info_attr += (sum_classes_dist / node.NUMBER_OF_EXAMPLES_SEEN) * expected_info_v
        node.CANDIDATE_ATTRIBUTES_SCORES[attr] = expected_info_tr - expected_info_attr


# HERE WE GO WITH THE "HOEFFDING NODE".
class HoeffdingNode:

    def __init__(self, classes, candidate_attributes):
        # CREATING ATTRIBUTES
        self.__ATTRIBUTE_NAME = None
        self.NUMBER_OF_EXAMPLES_SEEN = 0

        self.CLASSES_DISTRIBUTIONS = OrderedDict()
        self.CLASSES_PROB_DISTRIBUTIONS = OrderedDict()

        self.CANDIDATE_ATTRIBUTES = candidate_attributes

        self.CANDIDATE_ATTRIBUTES_VALUES_DISTRIBUTIONS = OrderedDict()
        self.CANDIDATE_ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS = OrderedDict()
        self.CANDIDATE_ATTRIBUTES_SCORES = OrderedDict()

        self.PARENT = None
        self.BRANCHES = OrderedDict()
        self.__CLASS = None

        self.initialize_classes(classes)
        self.initialize_attributes(classes)

    def initialize_classes(self, classes):
        for c in classes:
            self.CLASSES_DISTRIBUTIONS[c] = 0
            self.CLASSES_PROB_DISTRIBUTIONS[c] = 0.0

    def initialize_attributes(self, classes):
        for attribute in self.CANDIDATE_ATTRIBUTES:
            self.CANDIDATE_ATTRIBUTES_VALUES_DISTRIBUTIONS[attribute.NAME] = OrderedDict()
            self.CANDIDATE_ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attribute.NAME] = OrderedDict()
            for value in attribute.POSSIBLE_VALUES:
                self.CANDIDATE_ATTRIBUTES_VALUES_DISTRIBUTIONS[attribute.NAME][value] = OrderedDict()
                self.CANDIDATE_ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attribute.NAME][value] = OrderedDict()
                for c in classes:
                    self.CANDIDATE_ATTRIBUTES_VALUES_DISTRIBUTIONS[attribute.NAME][value][c] = 0
                    self.CANDIDATE_ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attribute.NAME][value][c] = 0.0

    def set_attribute_name(self, name):
        """This function is called when an attribute has been considered as an appropriate choice of splitting!"""
        self.__ATTRIBUTE_NAME = name

    def get_attribute_name(self):
        return self.__ATTRIBUTE_NAME

    def set_class(self, c):
        """This function is called when the node is supposed to be labelled with the most frequent class."""
        self.__CLASS = c

    def get_class(self):
        c = self.__CLASS
        return c

    def get_child_node(self, value):
        return self.BRANCHES[value]


class HoeffdingTree(SuperClassifier):
    """This is the implementation of Hoeffding Tree which is also known as Very Fast Decision Tree (VFDT)
    in the literature. Hoeffding Tree is an incremental decision tree for particularly learning from data streams."""

    LEARNER_NAME = TornadoDic.HOEFFDING_TREE
    LEARNER_TYPE = TornadoDic.TRAINABLE
    LEARNER_CATEGORY = TornadoDic.NOM_CLASSIFIER

    def __init__(self, classes, attributes, delta=0.0000001, tie=0.05, n_min=200, leaf_prediction_mode=TornadoDic.NB,
                 max_memory_size=33554432, memory_check_step=1000000):

        super().__init__(classes, attributes)
        self.__ROOT = HoeffdingNode(classes, attributes)

        self.ATTRIBUTES_NAMES = []

        self.__DELTA = delta
        self.__TIE = tie
        self.__R = math.log2(len(classes))
        self.__N_min = n_min

        self.__MAX_MEMORY_SIZE = max_memory_size
        self.__MEMORY_CHECK_STEP = memory_check_step

        self.__PREDICTION_MODE = leaf_prediction_mode

        self.__set_attributes_names()

    def __set_attributes_names(self):
        for attribute in self.ATTRIBUTES:
            self.ATTRIBUTES_NAMES.append(attribute.NAME)

    def get_root(self):
        return self.__ROOT

    def __trace(self, instance):
        current_node = self.__ROOT
        while len(current_node.BRANCHES) != 0:
            index = self.ATTRIBUTES_NAMES.index(current_node.get_attribute_name())
            current_node = current_node.get_child_node(instance[index])
        return current_node

    def train(self, instance):

        x, y = instance[:-1], instance[-1]

        node = self.__trace(x)

        node.NUMBER_OF_EXAMPLES_SEEN += 1
        node.CLASSES_DISTRIBUTIONS[y] += 1

        if self.__PREDICTION_MODE == TornadoDic.NB:

            for c in self.CLASSES:
                node.CLASSES_PROB_DISTRIBUTIONS[c] = node.CLASSES_DISTRIBUTIONS[c] / node.NUMBER_OF_EXAMPLES_SEEN

            for i in range(0, len(x)):
                attribute, value = self.ATTRIBUTES[i], x[i]
                if node.CANDIDATE_ATTRIBUTES.__contains__(attribute):
                    node.CANDIDATE_ATTRIBUTES_VALUES_DISTRIBUTIONS[attribute.NAME][value][y] += 1

            for c, c_prob in node.CLASSES_DISTRIBUTIONS.items():
                for attr in node.CANDIDATE_ATTRIBUTES:
                    attr_name = attr.NAME
                    k = len(node.CANDIDATE_ATTRIBUTES_VALUES_DISTRIBUTIONS[attr_name])
                    for value in attr.POSSIBLE_VALUES:
                        d = node.CANDIDATE_ATTRIBUTES_VALUES_DISTRIBUTIONS[attr_name][value][c]
                        node.CANDIDATE_ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attr_name][value][c] = (d + 1) / (k + c_prob)

        most_populated_class = max(node.CLASSES_DISTRIBUTIONS.items(), key=operator.itemgetter(1))
        node.set_class(most_populated_class)

        if (node.NUMBER_OF_EXAMPLES_SEEN - 1) != most_populated_class[1]:
            calculate_info_gain(node)

        if node.NUMBER_OF_EXAMPLES_SEEN >= self.__N_min and len(node.CANDIDATE_ATTRIBUTES_SCORES) != 0:

            g1, g2 = self.__get_two_attributes_with_highest_scores(node.CANDIDATE_ATTRIBUTES_SCORES)
            epsilon = calculate_hoeffding_bound(self.__R, self.__DELTA, node.NUMBER_OF_EXAMPLES_SEEN)
            if g1[1] - g2[1] > epsilon or epsilon < self.__TIE:
                node.set_attribute_name(g1[0])
                new_attributes_set = self.ATTRIBUTES.copy()
                attribute_index = self.ATTRIBUTES_NAMES.index(g1[0])
                del new_attributes_set[attribute_index]
                for value in self.ATTRIBUTES[attribute_index].POSSIBLE_VALUES:
                    leaf = HoeffdingNode(self.CLASSES, new_attributes_set)
                    node.BRANCHES[value] = leaf
                    leaf.PARENT = node

    def print_tree(self, node, c=""):
        c += "\t"
        print(c + node.get_attribute_name() + " " + str(node.CLASSES_DISTRIBUTIONS))
        for branch, child in node.BRANCHES.items():
            print(c + ">" + branch + "<")
            if child.get_attribute_name() is not None:
                self.print_tree(child, c)
            else:
                print(c + str(child.PARENT.get_class()))

    def test(self, instance):
        if self._IS_READY:
            x = instance[0:len(instance) - 1]
            y = instance[len(instance) - 1]
            node = self.__trace(x)

            if node.get_class() is None:
                node = node.PARENT

            if self.__PREDICTION_MODE == TornadoDic.MC:
                prediction = node.get_class()[0]
            else:
                predictions = OrderedDict()
                for c in node.CLASSES_DISTRIBUTIONS.keys():
                    pr = node.CLASSES_PROB_DISTRIBUTIONS[c]
                    for attr_index in range(0, len(x)):
                        if node.CANDIDATE_ATTRIBUTES.__contains__(self.ATTRIBUTES[attr_index]):
                            attr = self.ATTRIBUTES[attr_index]
                            value = x[attr_index]
                            pr *= node.CANDIDATE_ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attr.NAME][value][c]
                    predictions[c] = pr
                prediction = max(predictions.items(), key=operator.itemgetter(1))[0]

            self.update_confusion_matrix(y, prediction)

            return prediction

        else:
            print("Please train a Hoeffding Tree classifier first.")
            exit()

    def get_prediction_prob(self, X):

        node = self.__trace(X)

        if node.get_class() is None:
            node = node.PARENT

        prob = []
        if self.__PREDICTION_MODE == TornadoDic.MC:
            for c in self.CLASSES:
                prob.append(node.CLASSES_PROB_DISTRIBUTIONS[c])
        else:
            for c in self.CLASSES:
                pr = node.CLASSES_PROB_DISTRIBUTIONS[c]
                for attr_index in range(0, len(X)):
                    if node.CANDIDATE_ATTRIBUTES.__contains__(self.ATTRIBUTES[attr_index]):
                        attr = self.ATTRIBUTES[attr_index]
                        value = X[attr_index]
                        pr *= node.CANDIDATE_ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attr.NAME][value][c]
                prob.append(pr)

            prob_sum = sum(prob)
            if prob_sum != 0.0:
                prob = [x / prob_sum for x in prob]
            else:
                prob = [0.0 for x in prob]

        pred_prob = {}
        for i, c in enumerate(self.CLASSES):
            pred_prob[c] = prob[i]

        return pred_prob

    @staticmethod
    def __get_two_attributes_with_highest_scores(attributes_scores):
        sorted_attributes_scores = sorted(attributes_scores.items(), key=operator.itemgetter(1), reverse=True)
        g1 = sorted_attributes_scores[0]
        if len(sorted_attributes_scores) >= 2:
            g2 = sorted_attributes_scores[1]
        else:
            g2 = (0, 0)
        return g1, g2

    def reset(self):
        super()._reset_stats()
        del self.__ROOT
        gc.collect()
        self.__ROOT = HoeffdingNode(self.CLASSES, self.ATTRIBUTES)
