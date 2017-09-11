"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Stacking Fast Hoeffding Drift Detection Method (FHDDMS) Implementation ***
Paper: Reservoir of Diverse Adaptive Learners and Stacking Fast Hoeffding Drift Detection Methods for Evolving Data Streams
URL: https://arxiv.org/pdf/1709.02457.pdf
"""

import math

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class FHDDMS(SuperDetector):
    """The Stacking Fast Hoeffding Drift Detection Method (FHDDMS) class."""

    DETECTOR_NAME = TornadoDic.FHDDMS

    def __init__(self, m=4, n=25, delta=0.000001):

        super().__init__()

        self._WIN = []
        self._WIN_SIZE = m * n

        self._S_WIN_NUM = m
        self._S_WIN_SIZE = n
        self._DELTA = delta

        self._mu_max_short = 0
        self._mu_max_large = 0

    def __cal_hoeffding_bound(self, n):
        return math.sqrt(math.log((1 / self._DELTA), math.e) / (2 * n))

    def run(self, pr):
        drift_status = False
        warning_status = False

        if len(self._WIN) >= self._WIN_SIZE:
            self._WIN.pop(0)
        self._WIN.append(pr)

        if len(self._WIN) == self._WIN_SIZE:
            # TESTING THE SHORT WINDOW
            sub_wins_mu = []
            for i in range(0, self._S_WIN_NUM):
                sub_win = self._WIN[i * self._S_WIN_SIZE: (i + 1) * self._S_WIN_SIZE]
                sub_wins_mu.append(sub_win.count(True) / len(sub_win))
            if self._mu_max_short < sub_wins_mu[self._S_WIN_NUM - 1]:
                self._mu_max_short = sub_wins_mu[self._S_WIN_NUM - 1]
            if self._mu_max_short - sub_wins_mu[self._S_WIN_NUM - 1] > self.__cal_hoeffding_bound(self._S_WIN_SIZE):
                return False, True

            # TESTING THE LONG WINDOW
            mu_long = sum(sub_wins_mu) / self._S_WIN_NUM
            if self._mu_max_large < mu_long:
                self._mu_max_large = mu_long
            if self._mu_max_large - mu_long > self.__cal_hoeffding_bound(self._WIN_SIZE):
                return False, True

        return warning_status, drift_status

    def reset(self):
        super().reset()
        self._WIN.clear()
        self._mu_max_short = 0
        self._mu_max_large = 0

    def get_settings(self):
        return [str(self._S_WIN_SIZE) + "." + str(self._WIN_SIZE) + "." + str(self._DELTA),
                "$n_s$:" + str(self._S_WIN_SIZE) + ", " +
                "$n_l$:" + str(self._WIN_SIZE) + ", " +
                "$\delta$:" + str(self._DELTA).upper()]
