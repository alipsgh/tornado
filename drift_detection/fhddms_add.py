"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Additive Stacking Fast Hoeffding Drift Detection Method (FHDDMS.add) Implementation ***
Paper: Reservoir of Diverse Adaptive Learners and Stacking Fast Hoeffding Drift Detection Methods for Evolving Data Streams
URL: https://arxiv.org/pdf/1709.02457.pdf
"""

import math

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class FHDDMS_add(SuperDetector):
    """The Additive Stacking Fast Hoeffding Drift Detection Method (FHDDMS.add) class."""

    DETECTOR_NAME = TornadoDic.FHDDMS_add

    def __init__(self, m=4, n=25, delta=0.000001):

        super().__init__()

        self._ELEMENT_SIZE = n
        self._DELTA = delta

        self._stack = []
        self.init_stack(m)

        self._first_round = True
        self._counter = 0
        self._mu_max_short = 0
        self._mu_max_large = 0
        self._num_ones = 0

    def init_stack(self, size):
        self._stack.clear()
        for i in range(0, size):
            self._stack.append(0.0)

    def __cal_hoeffding_bound(self, n):
        return math.sqrt(math.log((1 / self._DELTA), math.e) / (2 * n))

    def run(self, pr):
        drift_status = False
        warning_status = False

        self._counter += 1

        if self._counter == (len(self._stack) * self._ELEMENT_SIZE) + 1:
            self._counter -= self._ELEMENT_SIZE
            self._num_ones -= self._stack[0]
            self._stack.pop(0)
            self._stack.append(0.0)
            if self._first_round is True:
                self._first_round = False

        if self._first_round is True:
            index = int(self._counter / self._ELEMENT_SIZE)
            if index == len(self._stack):
                index -= 1
        else:
            index = len(self._stack) - 1

        if pr is True:
            self._stack[index] += 1
            self._num_ones += 1

        # TESTING THE NEW SUB-WINDOWS
        if self._counter % self._ELEMENT_SIZE == 0:
            m_temp = self._stack[index] / self._ELEMENT_SIZE
            if self._mu_max_short < m_temp:
                self._mu_max_short = m_temp
            if self._mu_max_short - m_temp > self.__cal_hoeffding_bound(self._ELEMENT_SIZE):
                return False, True

        # TESTING THE WHOLE WINDOW
        if self._counter == len(self._stack) * self._ELEMENT_SIZE:
            m_temp = self._num_ones / (len(self._stack) * self._ELEMENT_SIZE)
            if self._mu_max_large < m_temp:
                self._mu_max_large = m_temp
            if self._mu_max_large - m_temp > self.__cal_hoeffding_bound(len(self._stack) * self._ELEMENT_SIZE):
                return False, True

        return warning_status, drift_status

    def reset(self):
        super().reset()
        self.init_stack(len(self._stack))
        self._first_round = True
        self._counter = 0
        self._mu_max_short = 0
        self._mu_max_large = 0
        self._num_ones = 0

    def get_settings(self):
        return [str(len(self._stack)) + "." + str(self._ELEMENT_SIZE) + "." + str(self._DELTA),
                "$n_s$:" + str(self._ELEMENT_SIZE) + ", " +
                "$n_l$:" + str(len(self._stack) * self._ELEMENT_SIZE) + ", " +
                "$\delta$:" + str(self._DELTA).upper()]
