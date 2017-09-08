"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Perceptron Implementation ***
Paper: Bifet, Albert, et al. "Fast perceptron decision tree learning from evolving data streams."
Published in: Advances in knowledge discovery and data mining (2010): 299-310.
URL: http://www.cs.waikato.ac.nz/~eibe/pubs/Perceptron.pdf
"""

import math
import operator
import random
from collections import OrderedDict

from classifier.classifier import SuperClassifier
from data_structures.attribute import Attribute
from dictionary.tornado_dictionary import TornadoDic


class Perceptron(SuperClassifier):
    """This is the implementation of a single perceptron for learning from data streams."""

    LEARNER_NAME = TornadoDic.PERCEPTRON
    LEARNER_TYPE = TornadoDic.TRAINABLE
    LEARNER_CATEGORY = TornadoDic.NUM_CLASSIFIER

    __BIAS_ATTRIBUTE = Attribute()
    __BIAS_ATTRIBUTE.set_name("bias")
    __BIAS_ATTRIBUTE.set_type(TornadoDic.NUMERIC_ATTRIBUTE)
    __BIAS_ATTRIBUTE.set_possible_values(1)

    def __init__(self, labels, attributes, learning_rate=1):
        super().__init__(labels, attributes)

        attributes.append(self.__BIAS_ATTRIBUTE)
        self.WEIGHTS = OrderedDict()
        self.__initialize_weights()
        self.LEARNING_RATE = learning_rate

    def __initialize_weights(self):
        for c in self.CLASSES:
            self.WEIGHTS[c] = OrderedDict()
            for a in self.ATTRIBUTES:
                self.WEIGHTS[c][a.NAME] = 0.2 * random.random() - 0.1

    def train(self, instance):
        x = instance[0:len(instance) - 1]
        x.append(1)
        y_real = instance[len(instance) - 1]
        predictions = OrderedDict()
        for c in self.CLASSES:
            predictions[c] = self.predict(x, c)

        for c in self.CLASSES:
            actual = 1 if c == y_real else 0
            delta = (actual - predictions[c]) * predictions[c] * (1 - predictions[c])
            for i in range(0, len(instance)):
                self.WEIGHTS[c][self.ATTRIBUTES[i].NAME] += self.LEARNING_RATE * delta * x[i]
        self._IS_READY = True

    def predict(self, x, c):
        s = 0
        for i in range(0, len(x)):
            s += self.WEIGHTS[c][self.ATTRIBUTES[i].NAME] * x[i]
        p = 1 / (1 + math.exp(-s))
        return p

    def test(self, instance):
        if self._IS_READY:
            x = instance[0:len(instance) - 1]
            y = instance[len(instance) - 1]
            x.append(1)
            predictions = OrderedDict()
            for c in list(self.CLASSES):
                predictions[c] = self.predict(x, c)
            y_predicted = max(predictions.items(), key=operator.itemgetter(1))[0]
            self.update_confusion_matrix(y, y_predicted)
            return y_predicted
        else:
            print("Please train a Perceptron classifier first!")
            exit()

    def reset(self):
        super()._reset_stats()
        self.WEIGHTS = OrderedDict()
        self.__initialize_weights()
