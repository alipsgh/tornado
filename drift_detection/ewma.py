"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Exponentially Weighted Moving Average (EWMA) Method Implementation ***
Paper: Ross, Gordon J., et al. "Exponentially weighted moving average charts for detecting concept drift."
Published in: Pattern Recognition Letters 33.2 (2012): 191-198.
URL: https://arxiv.org/pdf/1212.6018.pdf
"""

import math

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class EWMA(SuperDetector):
    """The Exponentially Weighted Moving Average (EWMA) drift detection method class."""

    DETECTOR_NAME = TornadoDic.EWMA

    def __init__(self, min_instance=30, lambda_=0.2):

        super().__init__()

        self.MINIMUM_NUM_INSTANCES = min_instance

        self.m_n = 1.0
        self.m_sum = 0.0
        self.m_p = 0.0
        self.m_s = 0.0
        self.z_t = 0.0
        self.lambda_ = lambda_

    def run(self, pr):

        pr = 1 if pr is False else 0

        warning_status = False
        drift_status = False

        # 1. UPDATING STATS
        self.m_sum += pr
        self.m_p = self.m_sum / self.m_n
        self.m_s = math.sqrt(self.m_p * (1.0 - self.m_p) * self.lambda_ * (1.0 - math.pow(1.0 - self.lambda_, 2.0 * self.m_n)) / (2.0 - self.lambda_))
        self.m_n += 1

        self.z_t += self.lambda_ * (pr - self.z_t)
        L_t = 3.97 - 6.56 * self.m_p + 48.73 * math.pow(self.m_p, 3) - 330.13 * math.pow(self.m_p, 5) + 848.18 * math.pow(self.m_p, 7)

        # 2. UPDATING WARNING AND DRIFT STATUSES
        if self.m_n < self.MINIMUM_NUM_INSTANCES:
            return False, False

        if self.z_t > self.m_p + L_t * self.m_s:
            drift_status = True
        elif self.z_t > self.m_p + 0.5 * L_t * self.m_s:
            warning_status = True

        return warning_status, drift_status

    def reset(self):
        super().reset()
        self.m_n = 1
        self.m_sum = 0
        self.m_p = 0
        self.m_s = 0
        self.z_t = 0

    def get_settings(self):
        return [str(self.MINIMUM_NUM_INSTANCES) + "." + str(self.lambda_),
                "$n_{min}$:" + str(self.MINIMUM_NUM_INSTANCES) + ", " +
                "$\lambda$:" + str(self.lambda_).upper()]
