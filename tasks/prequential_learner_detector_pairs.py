"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import random

import numpy
from pympler import asizeof

import copy

from archiver.archiver import Archiver
from evaluators.classifier_evaluator import PredictionEvaluator
from plotter.performance_plotter import *
from plotter.optimal_plotter import OptimalPairPlotter
from filters.score_processor import ScoreProcessor
from filters.attribute_handlers import *

# fp_level = 10
# fn_level = 2


class PrequentialMultiPairs:
    """This lets one run various pairs of (classifier, detector) against a data stream;
    and evaluate them prequentially, and calculated score of each pair."""

    def __init__(self, pairs, attributes, attributes_scheme,
                 actual_drift_points, drift_acceptance_interval, w_vec, project, color_set, legend_param=False):

        self.__instance_counter = 0
        self.__num_rubbish = 0

        self.pairs = pairs
        self.pairs_scores = []
        self.optimal_pair = []

        self.pairs_names = []
        self.unique_learners_names = []
        self.learners_stats = []
        self.detectors_stats = []
        self.pair_located_drift_points = []

        self.er = []
        self.dl_tp_fp_fn = []
        self.mu = []
        self.rt = []
        self.sc = []

        for pair in pairs:
            self.learners_stats.append([])
            self.detectors_stats.append([])
            self.pair_located_drift_points.append([])
            if legend_param is True:
                self.pairs_names.append(TornadoDic.get_short_names(pair[0].LEARNER_NAME) + " + " +
                                        pair[1].DETECTOR_NAME + "(" + pair[1].get_settings()[1] + ")")
            else:
                self.pairs_names.append(TornadoDic.get_short_names(pair[0].LEARNER_NAME) + " + " + pair[1].DETECTOR_NAME)
            if self.unique_learners_names.__contains__(pair[0].LEARNER_NAME) is False:
                self.unique_learners_names.append(pair[0].LEARNER_NAME)
            self.er.append([])
            self.mu.append([])
            self.rt.append([])
            self.sc.append([])

        self.actual_drift_points = actual_drift_points
        self.drift_acceptance_interval = drift_acceptance_interval
        self.w_vec = w_vec

        self.drift_loc_index = 0
        self.drift_current_context = 0

        self.__project_path = project.get_path()
        self.__project_name = project.get_name()

        self.attributes = attributes
        self.numeric_attribute_scheme = attributes_scheme['numeric']
        self.nominal_attribute_scheme = attributes_scheme['nominal']

        self.feedback_interval = 200
        self.feedback_counter = 0

        self.score_interval = 62
        self.score_counter = 0

        self.color_set = color_set

    def run(self, stream_records, random_seed=1):

        random.seed(random_seed)

        for record in stream_records:

            self.__instance_counter += 1

            if self.drift_loc_index < len(self.actual_drift_points) - 1:
                if self.__instance_counter > self.actual_drift_points[self.drift_loc_index] + self.drift_acceptance_interval:
                    self.drift_loc_index += 1

            if self.drift_current_context < len(self.actual_drift_points):
                if self.__instance_counter > self.actual_drift_points[self.drift_current_context]:
                    self.drift_current_context += 1

            percentage = (self.__instance_counter / len(stream_records)) * 100
            print("%0.2f" % percentage + "% of instances are processed!", end="\r")

            if record.__contains__("?"):
                self.__num_rubbish += 1
                continue

            for pair in self.pairs:
                learner = pair[0]
                detector = pair[1]
                index = self.pairs.index(pair)

                # ---------------------
                #  DATA TRANSFORMATION
                # ---------------------
                r = copy.copy(record)
                for k in range(0, len(r) - 1):
                    if learner.LEARNER_CATEGORY == TornadoDic.NOM_CLASSIFIER and self.attributes[k].TYPE == TornadoDic.NUMERIC_ATTRIBUTE:
                        r[k] = Discretizer.find_bin(r[k], self.nominal_attribute_scheme[k])
                    elif learner.LEARNER_CATEGORY == TornadoDic.NUM_CLASSIFIER and self.attributes[k].TYPE == TornadoDic.NOMINAL_ATTRIBUTE:
                        r[k] = NominalToNumericTransformer.map_attribute_value(r[k], self.numeric_attribute_scheme[k])
                # NORMALIZING NUMERIC DATA
                if learner.LEARNER_CATEGORY == TornadoDic.NUM_CLASSIFIER:
                    r[0:len(r) - 1] = Normalizer.normalize(r[0:len(r) - 1], self.numeric_attribute_scheme)

                # ----------------------
                #  PREQUENTIAL LEARNING
                # ----------------------
                if learner.is_ready():
                    real_class = r[len(r) - 1]
                    predicted_class = learner.do_testing(r)

                    prediction_status = True
                    if real_class != predicted_class:
                        prediction_status = False

                    # -----------------------
                    #  ANY DRIFTS DETECTED?
                    # -----------------------
                    warning_status, drift_status = detector.detect(prediction_status)
                    if drift_status:

                        # APPEND 1 INTO LOCATED DRIFT POINTS
                        self.pair_located_drift_points[index].append(1)

                        # APPENDING ERROR-RATE, MEMORY USAGE, AND RUNTIME OF CLASSIFIER
                        learner_error_rate = PredictionEvaluator.calculate(TornadoDic.ERROR_RATE, learner.get_confusion_matrix())
                        learner_error_rate = round(learner_error_rate, 4)
                        learner_runtime = learner.get_running_time()
                        learner_mem_use = asizeof.asizeof(learner, limit=20) / 1000
                        self.learners_stats[index].append([learner_error_rate, learner_mem_use, learner_runtime])

                        # APPENDING FP, FN, MEMORY USAGE, AND RUNTIME OF DETECTOR
                        delay, [tp_loc, tp], fp, fn, mem, runtime = self.detectors_stats[index][len(self.detectors_stats[index]) - 1]
                        actual_drift_loc = self.actual_drift_points[self.drift_loc_index]
                        if actual_drift_loc <= self.__instance_counter <= actual_drift_loc + self.drift_acceptance_interval:
                            if self.__instance_counter - tp_loc < self.drift_acceptance_interval:
                                fp += 1
                            else:
                                tp += 1
                                tp_loc = self.__instance_counter
                        else:
                            fp += 1
                        mem = asizeof.asizeof(detector) / 1000
                        runtime = detector.RUNTIME
                        self.detectors_stats[index].append([delay, [tp_loc, tp], fp, fn, mem, runtime])

                        learner.reset()
                        detector.reset()
                        continue

                    if learner.LEARNER_TYPE == TornadoDic.TRAINABLE:
                        learner.do_training(r)
                    else:
                        learner.do_loading(r)
                else:
                    if learner.LEARNER_TYPE == TornadoDic.TRAINABLE:
                        learner.do_training(r)
                    else:
                        learner.do_loading(r)

                    learner.set_ready()
                    learner.update_confusion_matrix(r[len(r) - 1], r[len(r) - 1])

                self.pair_located_drift_points[index].append(0)

                # APPENDING ERROR-RATE, MEMORY USAGE, AND RUNTIME OF CLASSIFIERS
                learner_error_rate = PredictionEvaluator.calculate(TornadoDic.ERROR_RATE, learner.get_confusion_matrix())
                learner_error_rate = round(learner_error_rate, 4)
                if self.feedback_counter % self.feedback_interval == 0 or self.__instance_counter == len(stream_records):
                    learner_mem_use = asizeof.asizeof(learner, limit=20) / 1000
                else:
                    learner_mem_use = self.learners_stats[index][len(self.learners_stats[index]) - 1][1]
                learner_runtime = learner.get_running_time()
                self.learners_stats[index].append([learner_error_rate, learner_mem_use, learner_runtime])

                # APPENDING FP, FN, MEMORY USAGE, AND RUNTIME OF DRIFT DETECTORS
                if self.__instance_counter == 1:
                    delay, [tp_loc, tp], fp, fn, mem, runtime = [0, [0, 0], 0, 0, 0, 0]
                else:
                    delay, [tp_loc, tp], fp, fn, mem, runtime = self.detectors_stats[index][len(self.detectors_stats[index]) - 1]
                    runtime = detector.RUNTIME
                    # print(runtime)
                    if self.feedback_counter % self.feedback_interval == 0 or self.__instance_counter == len(stream_records):
                        mem = asizeof.asizeof(detector) / 1000
                    if self.drift_current_context >= 1:
                        if self.__instance_counter >= self.actual_drift_points[self.drift_current_context - 1]:
                            fn = self.drift_current_context - tp
                            if self.__instance_counter <= self.actual_drift_points[self.drift_current_context - 1] + self.drift_acceptance_interval:
                                if tp_loc < self.actual_drift_points[self.drift_current_context - 1] or tp_loc > self.actual_drift_points[self.drift_current_context - 1] + self.drift_acceptance_interval:
                                    delay += 1
                self.detectors_stats[index].append([delay, [tp_loc, tp], fp, fn, mem, runtime])
                # print(instance_counter, detectors_stats[index][len(detectors_stats[index]) - 1])

            # CALCULATE SCORES & OPTIMAL CHOICE
            if self.score_counter % self.score_interval == 0:

                current_stats = []
                for i in range(0, len(self.pairs)):
                    ce, cm, cr = self.learners_stats[i][len(self.learners_stats[i]) - 1]
                    dd, [dtp_loc, dtp], dfp, dfn, dm, dr = self.detectors_stats[i][len(self.detectors_stats[i]) - 1]
                    current_stats.append([ce, dd, dfp, dfn, cm + dm, cr + dr])

                # current_stats = ScoreProcessor.penalize_high_dfp(fp_level, 2, 1, current_stats)
                # ranked_current_stats = ScoreProcessor.rank_matrix(current_stats)
                scaled_current_stats = ScoreProcessor.normalize_matrix(current_stats)
                scaled_current_scores = ScoreProcessor.calculate_weighted_scores(scaled_current_stats, self.w_vec)
                self.pairs_scores.append(scaled_current_scores)
                # print(scaled_current_scores)
                max_score = max(scaled_current_scores)
                indexes = numpy.argwhere(numpy.array(scaled_current_scores) == max_score).flatten().tolist()
                optimal_index = random.choice(indexes)
                # index = scaled_current_scores.index(max(scaled_current_scores))
                learner_name = self.pairs[optimal_index][0].LEARNER_NAME.upper()
                detector_name = self.pairs[optimal_index][1].DETECTOR_NAME.upper()
                optimal = learner_name + " + " + detector_name
                self.optimal_pair.append([optimal_index, optimal])
                # print(optimal)
                # for i in range(0, len(learners_detectors)):
                #    ce, cm, cr = learners_stats[i][len(learners_stats[i]) - 1]
                #    dd, [dtp_loc, dtp], dfp, dfn, dm, dr = detectors_stats[i][len(detectors_stats[i]) - 1]
                #    print("\t", learners_detectors_names[i], [ce, dd, dfp, dfn, cm + dm, cr + dr])

            self.feedback_counter += 1
            self.score_counter += 1

        self.store_stats()
        self.plot()
        self.archive()
        self.print_stats()

        print("THE END")
        print("\a")

    def store_stats(self):

        for i in range(0, len(self.er)):
            for j in range(0, len(self.learners_stats[i])):
                self.er[i].append(self.learners_stats[i][j][0])
                self.mu[i].append(self.learners_stats[i][j][1] + self.detectors_stats[i][j][4])
                self.rt[i].append(self.learners_stats[i][j][2] + self.detectors_stats[i][j][5])
            self.dl_tp_fp_fn.append(self.detectors_stats[i][len(self.detectors_stats[i]) - 1][0:4])

        for i in range(0, len(self.pairs_scores)):
            for j in range(0, len(self.sc)):
                self.sc[j].append(self.pairs_scores[i][j])

        stats_writer = open(self.__project_path + self.__project_name + ".txt", "w")
        stats_writer.write("[Name, Avg. Error-rate, Drift Detector Stats, Avg. Total Memory, Avg. Total Runtime, Avg. Score]" + "\n")
        for i in range(0, len(self.pairs)):
            name = self.pairs[i][0].LEARNER_NAME + " + " + self.pairs[i][1].DETECTOR_NAME
            ce = str(numpy.round(numpy.mean(self.er[i]), 4) * 100)
            cdm = str(numpy.round(numpy.mean(self.mu[i]), 2))
            cdr = str(numpy.round(numpy.mean(self.rt[i]), 2))
            ds = str(self.dl_tp_fp_fn[i])
            cds = str(numpy.round(numpy.mean(self.sc[i]), 2))
            stats_writer.write(name + ":\t" + ce + "\t" + ds + "\t" + cdm + "\t" + cdr + "\t" + cds + "\n")
        stats_writer.close()

    def plot(self):

        z_orders = []
        for i in range(0, len(self.pairs_names)):
            z_orders.append(len(self.pairs_names) - i + 1)

        file_name = self.__project_name + "_multi"

        # === Plotting Error-rates
        Plotter.plot_multiple(self.pairs_names, len(self.er[0]), self.er, 'Error-rate', self.__project_name,
                              self.__project_path, file_name, [0.0, 1], (1, 1.0125), 2,
                              len(self.unique_learners_names), 313, self.color_set, z_orders, print_legend=True)

        # === Plotting Memory Usage
        Plotter.plot_multiple(self.pairs_names, len(self.mu[0]), self.mu, 'Memory Usage (Kilobytes)', self.__project_name,
                              self.__project_path, file_name, [0.0, 150], (1, 1.01225), 2,
                              len(self.unique_learners_names), 313, self.color_set, z_orders, print_legend=True)

        # === Plotting Runtime
        Plotter.plot_multiple(self.pairs_names, len(self.rt[0]), self.rt, 'Runtime (Milliseconds)', self.__project_name,
                              self.__project_path, file_name, [0.0, 1000], (1, 1.01225), 2,
                              len(self.unique_learners_names), 313, self.color_set, z_orders, print_legend=True)

        # === Plotting Scores
        Plotter.plot_multiple(self.pairs_names, len(self.sc[0]), self.sc, 'Score', self.__project_name,
                              self.__project_path, file_name, [0.0, 1.025], (1, 1.01225), 2,
                              len(self.unique_learners_names), 14, self.color_set, z_orders, print_legend=True)

        # === Plotting Drift Points
        Plotter.plot_multi_ddms_points(self.pairs_names, self.pair_located_drift_points,
                                       self.__project_name, self.__project_path, self.__project_name, self.color_set)

        OptimalPairPlotter.plot_circles(self.optimal_pair, self.pairs_names, len(self.unique_learners_names),
                                        self.__project_name, self.__project_path, file_name,
                                        self.color_set, (1, 1.01225), 2, print_title=True, print_legend=True)

    def archive(self):

        Archiver.archive_multiple(self.pairs_names, self.er,
                                  self.__project_path, self.__project_name, 'Error-rate')
        Archiver.archive_multiple(self.pairs_names, self.mu,
                                  self.__project_path, self.__project_name, 'Memory Usage (Kilobytes)')
        Archiver.archive_multiple(self.pairs_names, self.rt,
                                  self.__project_path, self.__project_name, 'Runtime (Milliseconds)')
        Archiver.archive_multiple(self.pairs_names, self.sc,
                                  self.__project_path, self.__project_name, 'Score')
        Archiver.archive_multiple(self.pairs_names, self.learners_stats,
                                  self.__project_path, self.__project_name, 'learners_stats')
        Archiver.archive_multiple(self.pairs_names, self.detectors_stats,
                                  self.__project_path, self.__project_name, 'detectors_stats')

    def print_stats(self):

        for learner_detector in self.pairs_names:
            index = self.pairs_names.index(learner_detector)
            learner_stats = self.learners_stats[index][len(self.learners_stats[index]) - 1]
            detector_stats = self.detectors_stats[index][len(self.detectors_stats[index]) - 1]
            print(learner_detector, learner_stats, detector_stats)

