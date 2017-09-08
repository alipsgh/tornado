"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Hoeffding's Bound based Drift Detection Method - W_test Scheme Implementation (HDDM.W.test) ***
Paper: Frías-Blanco, Isvani, et al. "Online and non-parametric drift detection methods based on Hoeffding’s bounds."
Published in: IEEE Transactions on Knowledge and Data Engineering 27.3 (2015): 810-823.
URL: http://ieeexplore.ieee.org/abstract/document/6871418/
"""

import math
import sys

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class SampleInfo:

    def __init__(self):
        self.EWMA_estimator = -1.0
        self.independent_bounded_condition_sum = 0.0


class HDDM_W_test(SuperDetector):
    """The Hoeffding's Bound based Drift Detection Method - W_test (HDDM.W.test) class."""

    DETECTOR_NAME = TornadoDic.HDDM_W_test

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005, lambda_=0.05, test_type='one-sided'):

        super().__init__()

        self.total = SampleInfo()
        self.sample1_decr_monitoring = SampleInfo()
        self.sample1_incr_monitoring = SampleInfo()
        self.sample2_decr_monitoring = SampleInfo()
        self.sample2_incr_monitoring = SampleInfo()
        self.incr_cut_point = sys.float_info.max
        self.decr_cut_point = sys.float_info.min

        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.lambda_ = lambda_
        self.test_type = test_type

    def run(self, pr):

        pr = 1.0 if pr is False else 0.0
        warning_status = False
        drift_status = False

        # 1. UPDATING STATS
        aux_decay_rate = 1 - self.lambda_
        if self.total.EWMA_estimator < 0.0:
            self.total.EWMA_estimator = pr
            self.total.independent_bounded_condition_sum = 1.0
        else:
            self.total.EWMA_estimator = self.lambda_ * pr + aux_decay_rate * self.total.EWMA_estimator
            self.total.independent_bounded_condition_sum = self.lambda_ * self.lambda_ + aux_decay_rate * aux_decay_rate * self.total.independent_bounded_condition_sum

        self.update_incr_statistics(pr)

        if self.monitor_mean_incr(self.drift_confidence):
            self.reset_parameters()
            warning_status = False
            drift_status = True
        elif self.monitor_mean_incr(self.warning_confidence):
            warning_status = True
            drift_status = False
        else:
            warning_status = False
            drift_status = False

        self.update_decr_statistics(pr)

        if self.test_type != 'one-sided' and self.monitor_mean_decr(self.drift_confidence):
            self.reset_parameters()

        return warning_status, drift_status

    def update_incr_statistics(self, pr):
        aux_decay = 1.0 - self.lambda_
        bound = math.sqrt(self.total.independent_bounded_condition_sum * math.log(1.0 / self.drift_confidence, math.e) / 2)

        if self.total.EWMA_estimator + bound < self.incr_cut_point:
            self.incr_cut_point = self.total.EWMA_estimator + bound
            self.sample1_incr_monitoring.EWMA_estimator = self.total.EWMA_estimator
            self.sample1_incr_monitoring.independent_bounded_condition_sum = self.total.independent_bounded_condition_sum
            self.sample2_incr_monitoring = SampleInfo()
        else:
            if self.sample2_incr_monitoring.EWMA_estimator < 0.0:
                self.sample2_incr_monitoring.EWMA_estimator = pr
                self.sample2_incr_monitoring.independent_bounded_condition_sum = 1.0
            else:
                self.sample2_incr_monitoring.EWMA_estimator = self.lambda_ * pr + aux_decay * self.sample2_incr_monitoring.EWMA_estimator
                self.sample2_incr_monitoring.independent_bounded_condition_sum = self.lambda_ * self.lambda_ + aux_decay * aux_decay * self.sample2_incr_monitoring.independent_bounded_condition_sum

    def monitor_mean_incr(self, confidence_level):
        return self.detect_mean_increment(self.sample1_incr_monitoring, self.sample2_incr_monitoring, confidence_level)

    @staticmethod
    def detect_mean_increment(sample_1, sample_2, confidence_level):
        if sample_1.EWMA_estimator < 0.0 or sample_2.EWMA_estimator < 0.0:
            return False
        bound = math.sqrt((sample_1.independent_bounded_condition_sum + sample_2.independent_bounded_condition_sum) * math.log(1 / confidence_level, math.e) / 2)
        return sample_2.EWMA_estimator - sample_1.EWMA_estimator > bound

    def update_decr_statistics(self, pr):
        aux_decay = 1.0 - self.lambda_
        epsilon = math.sqrt(self.total.independent_bounded_condition_sum * math.log(1.0 / self.drift_confidence, math.e) / 2)

        if self.total.EWMA_estimator - epsilon > self.decr_cut_point:
            self.decr_cut_point = self.total.EWMA_estimator - epsilon
            self.sample1_decr_monitoring.EWMA_estimator = self.total.EWMA_estimator
            self.sample1_decr_monitoring.independent_bounded_condition_sum = self.total.independent_bounded_condition_sum
            self.sample2_decr_monitoring = SampleInfo()
        else:
            if self.sample2_decr_monitoring.EWMA_estimator < 0.0:
                self.sample2_decr_monitoring.EWMA_estimator = pr
                self.sample2_decr_monitoring.independent_bounded_condition_sum = 1.0
            else:
                self.sample2_decr_monitoring.EWMA_estimator = self.lambda_ * pr + aux_decay * self.sample2_decr_monitoring.EWMA_estimator
                self.sample2_decr_monitoring.independent_bounded_condition_sum = self.lambda_ * self.lambda_ + aux_decay * aux_decay * self.sample2_decr_monitoring.independent_bounded_condition_sum

    def monitor_mean_decr(self, confidence_level):
        return self.detect_mean_increment(self.sample2_decr_monitoring, self.sample1_decr_monitoring, confidence_level)

    def reset_parameters(self):
        self.total = SampleInfo()
        self.sample1_decr_monitoring = SampleInfo()
        self.sample1_incr_monitoring = SampleInfo()
        self.sample2_decr_monitoring = SampleInfo()
        self.sample2_incr_monitoring = SampleInfo()
        self.incr_cut_point = sys.float_info.max
        self.decr_cut_point = sys.float_info.min

    def reset(self):
        super().reset()
        self.reset_parameters()

    def get_settings(self):
        return [str(self.drift_confidence) + "." + str(self.warning_confidence) + "." +
                str(self.lambda_) + "." + str(self.test_type),
                "$\delta_d$:" + str(self.drift_confidence).upper() + ", " +
                "$\delta_w$:" + str(self.warning_confidence).upper() + ", " +
                "$\lambda$:" + str(self.lambda_).upper()]
