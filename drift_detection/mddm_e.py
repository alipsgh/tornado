"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The McDiarmid Drift Detection Method - Euler Scheme (MDDM_E) Implementation ***
Paper: Pesaranghader, Ali, et al. "McDiarmid Drift Detection Method for Evolving Data Streams."
Published in: International Joint Conference on Neural Network (IJCNN 2018)
URL: https://arxiv.org/abs/1710.02030
"""

import math

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class MDDM_E(SuperDetector):
    """The McDiarmid Drift Detection Method - Euler Scheme (MDDM_E) class."""

    DETECTOR_NAME = TornadoDic.MDDM_E

    def __init__(self, n=100, lambda_=0.01, delta=0.000001):

        super().__init__()

        self.win = []
        self.n = n
        self.lambda_ = lambda_
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
        sum_, bound_sum, r, ratio = 0, 0, 1, math.pow(math.e, self.lambda_)
        for i in range(self.n):
            sum_ += r
            r *= ratio
        r = 1
        for i in range(self.n):
            bound_sum += math.pow(r / sum_, 2)
            r *= ratio
        return bound_sum

    def cal_w_sigma(self):
        total_sum, win_sum, r, ratio = 0, 0, 1, math.pow(math.e, self.lambda_)
        for i in range(self.n):
            total_sum += r
            win_sum += self.win[i] * r
            r *= ratio
        return win_sum / total_sum

    def get_settings(self):
        settings = [str(self.n) + "." + str(self.delta),
                    "$n$:" + str(self.n) + ", " +
                    "$l$:" + str(self.lambda_) + ", " +
                    "$\delta$:" + str(self.delta).upper()]
        return settings
