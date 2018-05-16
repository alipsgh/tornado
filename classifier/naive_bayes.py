"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import operator
from collections import OrderedDict

from classifier.classifier import SuperClassifier
from dictionary.tornado_dictionary import TornadoDic


class NaiveBayes(SuperClassifier):
    """This is the implementation of incremental naive bayes classifier for learning from data streams."""

    LEARNER_NAME = TornadoDic.NAIVE_BAYES
    LEARNER_TYPE = TornadoDic.TRAINABLE
    LEARNER_CATEGORY = TornadoDic.NOM_CLASSIFIER

    def __init__(self, labels, attributes, smoothing_parameter=1):

        super().__init__(labels, attributes)

        self.ATTRIBUTES_NAMES = []
        self.ALPHA = smoothing_parameter

        self.CLASSES_DISTRIBUTIONS = OrderedDict()
        self.CLASSES_PROB_DISTRIBUTIONS = OrderedDict()

        self.ATTRIBUTES_VALUES_DISTRIBUTIONS = OrderedDict()
        self.ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS = OrderedDict()

        self.__initialize_classes()
        self.__initialize_attributes()

    def __initialize_classes(self):
        for c in self.CLASSES:
            self.CLASSES_DISTRIBUTIONS[c] = 0
            self.CLASSES_PROB_DISTRIBUTIONS[c] = 0.0

    def __initialize_attributes(self):
        for attr in self.ATTRIBUTES:
            self.ATTRIBUTES_NAMES.append(attr.NAME)
            self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr.NAME] = OrderedDict()
            self.ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attr.NAME] = OrderedDict()
            for v in attr.POSSIBLE_VALUES:
                self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr.NAME][v] = OrderedDict()
                self.ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attr.NAME][v] = OrderedDict()
                for c in self.CLASSES:
                    self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr.NAME][v][c] = 0
                    self.ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attr.NAME][v][c] = 0.0

    def train(self, instance):
        self.NUMBER_OF_INSTANCES_OBSERVED += 1
        self.__set_class_dist(instance)
        self.__set_attr_val_dist(instance)

    def __set_class_dist(self, instance):
        y = instance[len(instance) - 1]
        try:
            self.CLASSES_DISTRIBUTIONS[y] += 1
        except:
            print("ERROR")
        for y in self.CLASSES_DISTRIBUTIONS.keys():
            self.CLASSES_PROB_DISTRIBUTIONS[y] = self.CLASSES_DISTRIBUTIONS[y] / self.NUMBER_OF_INSTANCES_OBSERVED

    def get_classes_dist(self):
        return self.CLASSES_DISTRIBUTIONS

    def __set_attr_val_dist(self, instance):
        ln = len(instance)
        x = instance[0:ln - 1]
        y = instance[ln - 1]
        '''
        for attr_index in range(0, len(x)):
            attr = self.ATTRIBUTES_NAMES[attr_index]
            value = x[attr_index]
            self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr][value][y] += 1
            for c, c_dist in self.CLASSES_DISTRIBUTIONS.items():
                d = self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr][value][c]
                k = len(self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr])
                self.ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attr][value][c] = (d + 1) / (k + c_dist)
        '''
        for attr_index in range(0, len(x)):
            attr = self.ATTRIBUTES_NAMES[attr_index]
            value = x[attr_index]
            self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr][value][y] += 1

        for c, c_dist in self.CLASSES_DISTRIBUTIONS.items():
            for attr in self.ATTRIBUTES:
                attr_name = attr.NAME
                k = len(self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr_name])
                for value in attr.POSSIBLE_VALUES:
                    d = self.ATTRIBUTES_VALUES_DISTRIBUTIONS[attr_name][value][c]
                    self.ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attr_name][value][c] = (d + 1) / (k + c_dist)

    def test(self, instance):
        if self._IS_READY:
            predictions = OrderedDict()
            x = instance[0:len(instance) - 1]
            y = instance[len(instance) - 1]
            for c in self.CLASSES:
                pr = self.CLASSES_PROB_DISTRIBUTIONS[c]
                for attr_index in range(0, len(x)):
                    attr = self.ATTRIBUTES_NAMES[attr_index]
                    value = x[attr_index]
                    pr *= self.ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attr][value][c]
                predictions[c] = pr

            predicted_class = max(predictions.items(), key=operator.itemgetter(1))[0]
            self.update_confusion_matrix(y, predicted_class)
            return predicted_class
        else:
            print("Please train a Naive Bayes classifier first.")
            exit()

    def get_prediction_prob(self, X):

        prob = []
        for c in self.CLASSES:
            pr = self.CLASSES_PROB_DISTRIBUTIONS[c]
            for attr_index in range(0, len(X)):
                attr = self.ATTRIBUTES_NAMES[attr_index]
                value = X[attr_index]
                pr *= self.ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attr][value][c]
                if self.ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS[attr][value][c] == 0:
                    print("HI")
            prob.append(pr)

        '''
        prob_x = 0.0
        for i, c in enumerate(self.CLASSES):
            p_c = self.CLASSES_PROB_DISTRIBUTIONS[c]
            prob_x += (p_c * prob[i])
        prob = [x / prob_x for x in prob]
        '''

        prob_sum = sum(prob)
        if prob_sum != 0.0:
            prob = [x / prob_sum for x in prob]
        else:
            prob = [0.0 for x in prob]

        pred_prob = {}
        for i, c in enumerate(self.CLASSES):
            pred_prob[c] = prob[i]

        return pred_prob

    def reset(self):
        super()._reset_stats()
        self.CLASSES_DISTRIBUTIONS = OrderedDict()
        self.CLASSES_PROB_DISTRIBUTIONS = OrderedDict()
        self.ATTRIBUTES_NAMES = []
        self.ATTRIBUTES_VALUES_DISTRIBUTIONS = OrderedDict()
        self.ATTRIBUTES_VALUES_PROB_DISTRIBUTIONS = OrderedDict()
        self.__initialize_classes()
        self.__initialize_attributes()
