"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Reactive Drift Detection Method (RDDM) Implementation ***
Paper: Barros, Roberto, et al. "RDDM: Reactive drift detection method."
Published in: Expert Systems with Applications. Elsevier, 2017.
URL: https://www.sciencedirect.com/science/article/pii/S0957417417305614
"""

import math
import sys

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class RDDM(SuperDetector):
    """The Reactive Drift Detection Method (RDDM) class."""

    DETECTOR_NAME = TornadoDic.RDDM

    def __init__(self, min_instance=129,
                 warning_level=1.773,
                 drift_level=2.258,
                 max_size_concept=40000,
                 min_size_stable_concept=7000,
                 warn_limit=1400):

        super().__init__()

        self.min_num_instance = min_instance
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.max_concept_size = max_size_concept
        self.min_size_stable_concept = min_size_stable_concept
        self.warn_limit = warn_limit

        self.m_n = 1
        self.m_p = 1
        self.m_s = 0
        self.m_p_min = sys.maxsize
        self.m_s_min = sys.maxsize
        self.m_p_s_min = sys.maxsize

        self.stored_predictions = [0 for _ in range(self.min_size_stable_concept)]
        self.num_stored_instances = 0
        self.first_pos = 0
        self.last_pos = -1
        self.last_warn_pos = -1
        self.last_warn_inst = -1
        self.inst_num = 0
        self.rddm_drift = False
        self.is_change_detected = False
        self.is_warning_zone = False

    def run(self, pr):

        pr = 1 if pr is False else 0

        warning_status, drift_status = False, False

        if self.rddm_drift: #
            self.reset_rddm() #
            if self.last_warn_pos != -1: #
                self.first_pos = self.last_warn_pos #
                self.num_stored_instances = self.last_pos - self.first_pos + 1 #
                if self.num_stored_instances <= 0: #
                    self.num_stored_instances += self.min_size_stable_concept #

            pos = self.first_pos #
            for i in range(0, self.num_stored_instances): #
                self.m_p += ((self.stored_predictions[pos] - self.m_p) / self.m_n) #
                self.m_s = math.sqrt(self.m_p * (1 - self.m_p) / self.m_n)
                if self.is_change_detected and (self.m_n > self.min_num_instance) and (self.m_p + self.m_s < self.m_p_s_min):
                    self.m_p_min = self.m_p
                    self.m_s_min = self.m_s
                    self.m_p_s_min = self.m_p + self.m_s
                self.m_n += 1
                pos = (pos + 1) % self.min_size_stable_concept

            self.last_warn_pos = -1
            self.last_warn_inst = -1
            self.rddm_drift = False
            self.is_change_detected = False

        self.last_pos = (self.last_pos + 1) % self.min_size_stable_concept
        self.stored_predictions[self.last_pos] = pr
        if self.num_stored_instances < self.min_size_stable_concept:
            self.num_stored_instances += 1
        else:
            self.first_pos = (self.first_pos + 1) % self.min_size_stable_concept
            if self.last_warn_pos == self.last_pos:
                self.last_warn_pos = -1

        self.m_p += (pr - self.m_p) / self.m_n
        self.m_s = math.sqrt(self.m_p * (1 - self.m_p) / self.m_n)

        self.inst_num += 1
        self.m_n += 1
        self.is_warning_zone = False

        if self.m_n <= self.min_num_instance:
            return warning_status, drift_status

        if self.m_p + self.m_s < self.m_p_s_min:
            self.m_p_min = self.m_p
            self.m_s_min = self.m_s
            self.m_p_s_min = self.m_p + self.m_s

        if self.m_p + self.m_s > self.m_p_min + self.drift_level * self.m_s_min:
            self.is_change_detected, drift_status = True, True
            self.rddm_drift = True
            if self.last_warn_inst == -1:
                self.first_pos = self.last_pos
                self.num_stored_instances = 1
            return warning_status, drift_status

        if self.m_p + self.m_s > self.m_p_min + self.warning_level * self.m_s_min:
            if (self.last_warn_inst != -1) and (self.last_warn_inst + self.warn_limit <= self.inst_num):
                self.is_change_detected, drift_status = True, True
                self.rddm_drift = True
                self.first_pos = self.last_pos
                self.num_stored_instances = 1
                self.last_warn_pos = -1
                self.last_warn_inst = -1
                return warning_status, drift_status

            self.is_warning_zone, warning_status = True, True
            if self.last_warn_inst == -1:
                self.last_warn_inst = self.inst_num
                self.last_warn_pos = self.last_pos
        else:
            self.last_warn_inst = -1
            self.last_warn_pos = -1

        if self.m_n > self.max_concept_size and self.is_warning_zone is False:
            self.rddm_drift = True

        return warning_status, drift_status

    def reset_rddm(self):
        self.m_n = 1
        self.m_p = 1
        self.m_s = 0
        if self.is_change_detected:
            self.m_p_min = sys.maxsize
            self.m_s_min = sys.maxsize
            self.m_p_s_min = sys.maxsize

    def reset(self):
        super().reset()
        self.m_n = 1
        self.m_p = 1
        self.m_s = 0
        self.m_p_min = sys.maxsize
        self.m_s_min = sys.maxsize
        self.m_p_s_min = sys.maxsize

    def get_settings(self):
        return [str(self.min_num_instance),
                "$n_{min}$:" + str(self.min_num_instance)]
