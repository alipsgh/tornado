"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Drift Detection Method (DDM) Implementation ***
Paper: Gama, Joao, et al. "Learning with drift detection."
Published in: Brazilian Symposium on Artificial Intelligence. Springer, Berlin, Heidelberg, 2004.
URL: https://link.springer.com/chapter/10.1007/978-3-540-28645-5_29
"""

import math
import sys

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class DDM(SuperDetector):
    """The traditional Drift Detection Method (DDM) class."""

    DETECTOR_NAME = TornadoDic.DDM

    def __init__(self, min_instance=30):

        super().__init__()

        self.MINIMUM_NUM_INSTANCES = min_instance
        self.NUM_INSTANCES_SEEN = 1

        self.__P = 1
        self.__S = 0
        self.__P_min = sys.maxsize
        self.__S_min = sys.maxsize

    def run(self, pr):

        warning_status, drift_status = False, False

        pr = 1 if pr is False else 0

        # 1. UPDATING STATS
        self.__P += (pr - self.__P) / self.NUM_INSTANCES_SEEN
        self.__S = math.sqrt(self.__P * (1 - self.__P) / self.NUM_INSTANCES_SEEN)

        self.NUM_INSTANCES_SEEN += 1

        if self.NUM_INSTANCES_SEEN < self.MINIMUM_NUM_INSTANCES:
            return False, False

        if self.__P + self.__S <= self.__P_min + self.__S_min:
            self.__P_min = self.__P
            self.__S_min = self.__S

        # 2. UPDATING WARNING AND DRIFT STATUSES
        current_level = self.__P + self.__S
        warning_level = self.__P_min + 2 * self.__S_min
        drift_level = self.__P_min + 3 * self.__S_min

        if current_level > warning_level:
            warning_status = True

        if current_level > drift_level:
            drift_status = True

        return warning_status, drift_status

    def reset(self):
        super().reset()
        self.NUM_INSTANCES_SEEN = 1
        self.__P = 1
        self.__S = 0
        self.__P_min = sys.maxsize
        self.__S_min = sys.maxsize

    def get_settings(self):
        return [str(self.MINIMUM_NUM_INSTANCES),
                "$n_{min}$:" + str(self.MINIMUM_NUM_INSTANCES)]
