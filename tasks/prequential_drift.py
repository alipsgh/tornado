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


class PrequentialDrift:
    """This class lets one run a classifier with a drift detector against a data stream,
    and evaluate it prequentially over time."""

    def __init__(self, learner, drift_detector, attributes, attributes_scheme, project, memory_check_step=-1):

        self.learner = learner
        self.drift_detector = drift_detector

        self.__instance_counter = 0
        self.__num_rubbish = 0

        self.__learner_error_rate_array = []
        self.__learner_memory_usage = []
        self.__learner_runtime = []

        self.__located_drift_points = []
        self.__drift_detection_memory_usage = []
        self.__drift_detection_runtime = []

        self.__attributes = attributes
        self.__numeric_attribute_scheme = attributes_scheme['numeric']
        self.__nominal_attribute_scheme = attributes_scheme['nominal']

        self.__project_path = project.get_path()
        self.__project_name = project.get_name()

        self.__memory_check_step = memory_check_step

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

                prediction_status = True
                if real_class != predicted_class:
                    prediction_status = False

                # -----------------------
                #  Drift Detected?
                # -----------------------
                warning_status, drift_status = self.drift_detector.detect(prediction_status)
                if drift_status:
                    self.__located_drift_points.append(self.__instance_counter)
                    print("\n ->>> " + self.learner.LEARNER_NAME.title() + " faced a drift at instance " +
                          str(self.__instance_counter) + ".")
                    print("%0.2f" % percentage, " of instances are prequentially processed!", end="\r")

                    learner_error_rate = PredictionEvaluator.calculate(TornadoDic.ERROR_RATE,
                                                                       self.learner.get_global_confusion_matrix())
                    self.__learner_error_rate_array.append(round(learner_error_rate, 4))
                    self.__learner_memory_usage.append(asizeof.asizeof(self.learner, limit=20))
                    self.__learner_runtime.append(self.learner.get_running_time())

                    self.__drift_detection_memory_usage.append(asizeof.asizeof(self.drift_detector, limit=20))
                    self.__drift_detection_runtime.append(self.drift_detector.RUNTIME)

                    self.learner.reset()
                    self.drift_detector.reset()

                    continue

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

            if self.__memory_check_step != -1:
                if self.__instance_counter % self.__memory_check_step == 0:
                    self.__drift_detection_memory_usage.append(asizeof.asizeof(self.drift_detector, limit=20))

        print("\n" + "The stream is completely processed.")
        self.__store_stats()
        self.__plot()
        print("THE END!")
        print("\a")

    def __store_stats(self):

        learner_name = TornadoDic.get_short_names(self.learner.LEARNER_NAME)
        detector_name = self.drift_detector.DETECTOR_NAME
        detector_setting = self.drift_detector.get_settings()

        file_name = learner_name + "_" + detector_name + "." + detector_setting[0]
        st_wr = open(self.__project_path + file_name.lower() + ".txt", "w")

        lrn_error_rate = PredictionEvaluator.calculate_error_rate(self.learner.get_global_confusion_matrix())

        if len(self.__located_drift_points) != 0:
            # learner stats
            lrn_mem = numpy.mean(self.__learner_memory_usage)
            lrn_ave_runtime = numpy.mean(self.__learner_runtime)
            lrn_total_runtime = self.learner.get_total_running_time()
            # ddm stats
            ddm_mem = numpy.mean(self.__drift_detection_memory_usage)
            ddm_avg_runtime = numpy.mean(self.__drift_detection_runtime)
            ddm_total_runtime = self.drift_detector.TOTAL_RUNTIME
        else:
            lrn_mem = asizeof.asizeof(self.learner, limit=20)
            lrn_ave_runtime = self.learner.get_total_running_time()
            lrn_total_runtime = lrn_ave_runtime
            ddm_mem = asizeof.asizeof(self.drift_detector, limit=20)
            ddm_avg_runtime = self.drift_detector.TOTAL_RUNTIME
            ddm_total_runtime = ddm_avg_runtime

        stats = learner_name + " + " + detector_name + ": " + "\n\t" + \
                "Classifier Error-rate: " + "%0.2f" % (100 * lrn_error_rate) + "%" + "\n\t" + \
                "Classifier Average Memory Usage (bytes): " + "%0.2f" % lrn_mem + "\n\t" + \
                "Classifier Average Runtime (ms): " + "%0.2f" % lrn_ave_runtime + "\n\t" + \
                "Classifier Total Runtime (ms): " + "%0.2f" % lrn_total_runtime + "\n\t" + \
                "Average Detection Memory Usage (bytes): " + "%0.2f" % ddm_mem + "," + "\n\t" + \
                "Average Detection Runtime (ms): " + "%0.2f" % ddm_avg_runtime + "," + "\n\t" + \
                "Total Detection Runtime (ms): " + "%0.2f" % ddm_total_runtime + "," + "\n\t" + \
                "Error-rate: " + "%0.2f" % (100 * lrn_error_rate) + "\n\t" + \
                "Drift Points detected: " + str(self.__located_drift_points)

        print(stats)

        st_wr.write(stats)
        st_wr.close()

    def __plot(self):

        learner_name = TornadoDic.get_short_names(self.learner.LEARNER_NAME)
        detector_name = self.drift_detector.DETECTOR_NAME
        detector_setting = self.drift_detector.get_settings()
        file_name = learner_name + "_" + detector_name + "." + detector_setting[0]

        up_range = numpy.max(self.__learner_error_rate_array)
        up_range = 1 if up_range > 0.75 else round(up_range, 1) + 0.25

        pair_name = learner_name + ' + ' + detector_name + "(" + detector_setting[1] + ")"
        Plotter.plot_single(pair_name, self.__learner_error_rate_array, "Error-rate",
                            self.__project_name, self.__project_path, file_name, [0, up_range], 'upper right', 200)
        Archiver.archive_single(pair_name, self.__learner_error_rate_array,
                                self.__project_path, self.__project_name, 'Error-rate')

