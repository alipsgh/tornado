"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import copy
import random

import numpy
from pympler import asizeof

from archiver.archiver import Archiver
from evaluators.classifier_evaluator import PredictionEvaluator
from plotter.performance_plotter import *
from filters.attribute_handlers import *
from streams.readers.arff_reader import *


class Prequential:
    """This class lets one run a classifier against a data stream, and evaluate it prequentially over time."""

    def __init__(self, learner, attributes, attributes_scheme, project):

        self.learner = learner

        self.__instance_counter = 0
        self.__num_rubbish = 0

        self.__learner_error_rate_array = []

        self.__attributes = attributes
        self.__numeric_attribute_scheme = attributes_scheme['numeric']
        self.__nominal_attribute_scheme = attributes_scheme['nominal']

        self.__project_path = project.get_path()
        self.__project_name = project.get_name()

    def run(self, stream, random_seed=1):

        random.seed(random_seed)

        for record in stream:

            self.__instance_counter += 1

            percentage = (self.__instance_counter / len(stream)) * 100
            print("%0.2f" % percentage + "% of instances are prequentially processed!", end="\r")

            if record.__contains__("?"):
                self.__num_rubbish += 1
                continue

            # ---------------------
            #  Data Transformation
            # ---------------------
            r = copy.copy(record)
            for k in range(0, len(r) - 1):
                if self.learner.LEARNER_CATEGORY == TornadoDic.NOM_CLASSIFIER and self.__attributes[k].TYPE == TornadoDic.NUMERIC_ATTRIBUTE:
                    r[k] = Discretizer.find_bin(r[k], self.__nominal_attribute_scheme[k])
                elif self.learner.LEARNER_CATEGORY == TornadoDic.NUM_CLASSIFIER and self.__attributes[k].TYPE == TornadoDic.NOMINAL_ATTRIBUTE:
                    r[k] = NominalToNumericTransformer.map_attribute_value(r[k], self.__numeric_attribute_scheme[k])
            # NORMALIZING NUMERIC DATA
            if self.learner.LEARNER_CATEGORY == TornadoDic.NUM_CLASSIFIER:
                r[0:len(r) - 1] = Normalizer.normalize(r[0:len(r) - 1], self.__numeric_attribute_scheme)

            # ----------------------
            #  Prequential Learning
            # ----------------------
            if self.learner.is_ready():
                real_class = r[len(r) - 1]
                predicted_class = self.learner.do_testing(r)
                if self.learner.LEARNER_TYPE == TornadoDic.TRAINABLE:
                    self.learner.do_training(r)
                else:
                    self.learner.do_loading(r)
            else:
                if self.learner.LEARNER_TYPE == TornadoDic.TRAINABLE:
                    self.learner.do_training(r)
                else:
                    self.learner.do_loading(r)

                self.learner.set_ready()
                self.learner.update_confusion_matrix(r[len(r) - 1], r[len(r) - 1])

            learner_error_rate = PredictionEvaluator.calculate(TornadoDic.ERROR_RATE,
                                                               self.learner.get_confusion_matrix())
            learner_error_rate = round(learner_error_rate, 4)
            self.__learner_error_rate_array.append(learner_error_rate)

        print("\n" + "The stream is completely processed.")
        self.__store_stats()
        self.__plot()
        print("THE END!")
        print("\a")

    def __store_stats(self):

        st_wr = open(self.__project_path + TornadoDic.get_short_names(self.learner.LEARNER_NAME).lower() + ".txt", "w")

        lrn_error_rate = PredictionEvaluator.calculate_error_rate(self.learner.get_global_confusion_matrix())
        lrn_mem = asizeof.asizeof(self.learner, limit=20)
        lrn_runtime = self.learner.get_total_running_time()

        stats = self.learner.LEARNER_NAME + "\n\t" + \
                "Classifier Error-rate: " + "%0.2f" % (100 * lrn_error_rate) + "%" + "\n\t" + \
                "Classifier Memory Usage (bytes): " + "%0.2f" % lrn_mem + "\n\t" + \
                "Classifier Runtime (ms): " + "%0.2f" % lrn_runtime

        print(stats)

        st_wr.write(stats)
        st_wr.close()

    def __plot(self):

        file_name = TornadoDic.get_short_names(self.learner.LEARNER_NAME)
        pair_name = self.learner.LEARNER_NAME.title()

        up_range = numpy.max(self.__learner_error_rate_array)
        up_range = 1 if up_range > 0.75 else round(up_range, 1) + 0.25

        Plotter.plot_single(pair_name, self.__learner_error_rate_array, "Error-rate",
                            self.__project_name, self.__project_path, file_name, [0, up_range], 'upper right', 200)
        Archiver.archive_single(pair_name, self.__learner_error_rate_array,
                                self.__project_path, self.__project_name, 'Error-rate')

