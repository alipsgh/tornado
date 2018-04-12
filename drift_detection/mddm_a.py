"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The McDiarmid Drift Detection Method - Arithmetic Scheme (MDDM_A) Implementation ***
Paper: Pesaranghader, Ali, et al. "McDiarmid Drift Detection Method for Evolving Data Streams."
Published in: International Joint Conference on Neural Network (IJCNN 2018)
URL: https://arxiv.org/abs/1710.02030
"""

import math

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class MDDM_A(SuperDetector):
    """The McDiarmid Drift Detection Method - Arithmetic Scheme (MDDM_A) class."""

    DETECTOR_NAME = TornadoDic.MDDM_A

    def __init__(self, n=100, difference=0.01, delta=0.000001):

        super().__init__()

        self.win = []
        self.n = n
        self.difference = difference
        self.delta = delta

        self.e = math.sqrt(0.5 * self.cal_sigma() * (math.log(1 / self.delta, math.e)))
        self.u_max = 0

        self.DETECTOR_NAME += "." + str(n)

    def run(self, pr):

        drift_status = False

        if len(self.win) == self.n:
            self.win.pop(0)
        self.win.append(pr)

        if len(self.win) == self.n:
            u = self.cal_w_sigma()
            self.u_max = u if u > self.u_max else self.u_max
            drift_status = True if (self.u_max - u > self.e) else False

        return False, drift_status

    def reset(self):
        super().reset()
        self.win.clear()
        self.u_max = 0

    def cal_sigma(self):
        sum_, sigma = 0, 0
        for i in range(self.n):
            sum_ += (1 + i * self.difference)
        for i in range(self.n):
            sigma += math.pow((1 + i * self.difference) / sum_, 2)
        return sigma

    def cal_w_sigma(self):
        total_sum, win_sum = 0, 0
        for i in range(self.n):
            total_sum += 1 + i * self.difference
            win_sum += self.win[i] * (1 + i * self.difference)
        return win_sum / total_sum

    def get_settings(self):
        settings = [str(self.n) + "." + str(self.delta),
                    "$n$:" + str(self.n) + ", " +
                    "$d$:" + str(self.difference) + ", " +
                    "$\delta$:" + str(self.delta).upper()]
        return settings
