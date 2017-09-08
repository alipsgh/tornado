"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Hoeffding's Bound based Drift Detection Method - A_test Scheme Implementation (HDDM.A.test) ***
Paper: Frías-Blanco, Isvani, et al. "Online and non-parametric drift detection methods based on Hoeffding’s bounds."
Published in: IEEE Transactions on Knowledge and Data Engineering 27.3 (2015): 810-823.
URL: http://ieeexplore.ieee.org/abstract/document/6871418/
"""

import math

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class HDDM_A_test(SuperDetector):
    """The Hoeffding's Bound based Drift Detection Method - A_test (HDDM.A.test) class."""

    DETECTOR_NAME = TornadoDic.HDDM_A_test

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005, test_type='two-sided'):

        super().__init__()

        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.test_type = test_type

        self.n_min = 0
        self.c_min = 0
        self.total_n = 0
        self.total_c = 0
        self.n_max = 0
        self.c_max = 0

    def run(self, pr):

        pr = 1 if pr is False else 0

        warning_status = False
        drift_status = False

        # 1. UPDATING STATS
        self.total_n += 1
        self.total_c += pr

        if self.n_min == 0:
            self.n_min = self.total_n
            self.c_min = self.total_c

        if self.n_max == 0:
            self.n_max = self.total_n
            self.c_max = self.total_c

        cota = math.sqrt((1.0 / (2 * self.n_min)) * math.log(1.0 / self.drift_confidence, math.e))
        cota1 = math.sqrt((1.0 / (2 * self.total_n)) * math.log(1.0 / self.drift_confidence, math.e))
        if self.c_min / self.n_min + cota >= self.total_c / self.total_n + cota1:
            self.c_min = self.total_c
            self.n_min = self.total_n

        cota = math.sqrt((1.0 / (2 * self.n_max)) * math.log(1.0 / self.drift_confidence, math.e))
        if self.c_max / self.n_max - cota <= self.total_c / self.total_n - cota1:
            self.c_max = self.total_c
            self.n_max = self.total_n

        if self.mean_incr(self.drift_confidence):
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0
            drift_status = True
            warning_status = False
        elif self.mean_incr(self.warning_confidence):
            drift_status = False
            warning_status = True
        else:
            drift_status = False
            warning_status = False

        # 2. UPDATING WARNING AND DRIFT STATUSES
        if self.test_type == 'two-sided' and self.mean_decr():
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0

        return warning_status, drift_status

    def mean_incr(self, confidence_level):
        if self.n_min == self.total_n:
            return False
        m = (self.total_n - self.n_min) / self.n_min * (1.0 / self.total_n)
        cota = math.sqrt((m / 2) * math.log(2.0 / confidence_level, math.e))
        return self.total_c / self.total_n - self.c_min / self.n_min >= cota

    def mean_decr(self):
        if self.n_max == self.total_n:
            return False
        m = (self.total_n - self.n_max) / self.n_max * (1.0 / self.total_n)
        cota = math.sqrt((m / 2) * math.log(2.0 / self.drift_confidence, math.e))
        return self.c_max / self.n_max - self.total_c / self.total_n >= cota

    def reset(self):
        super().reset()
        self.n_min = 0
        self.c_min = 0
        self.total_n = 0
        self.total_c = 0
        self.n_max = 0
        self.c_max = 0

    def get_settings(self):
        return [str(self.drift_confidence) + "." + str(self.warning_confidence) + "." + str(self.test_type),
                "$\delta_d$:" + str(self.drift_confidence).upper() + ", " +
                "$\delta_w$:" + str(self.warning_confidence).upper()]
