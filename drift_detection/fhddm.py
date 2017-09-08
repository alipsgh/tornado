"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Fast Hoeffding Drift Detection Method (FHDDM) Implementation ***
Paper: Pesaranghader, Ali, and Herna L. Viktor. "Fast hoeffding drift detection method for evolving data streams."
Published in: Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer International Publishing, 2016.
URL: https://link.springer.com/chapter/10.1007/978-3-319-46227-1_7
"""

import math

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class FHDDM(SuperDetector):
    """The Fast Hoeffding Drift Detection Method (FHDDM) class."""

    DETECTOR_NAME = TornadoDic.FHDDM

    def __init__(self, n=100, delta=0.000001):

        super().__init__()

        self.__DELTA = delta
        self.__N = n
        self.__E = math.sqrt(math.log((1 / self.__DELTA), math.e) / (2 * self.__N))

        self.__WIN = []
        self.__MU_M = 0

    def run(self, pr):

        drift_status = False

        if len(self.__WIN) >= self.__N:
            self.__WIN.pop(0)
        self.__WIN.append(pr)

        if len(self.__WIN) >= self.__N:
            mu_t = self.__WIN.count(True) / self.__N
            if self.__MU_M < mu_t:
                self.__MU_M = mu_t
            drift_status = (self.__MU_M - mu_t) > self.__E

        return False, drift_status

    def reset(self):
        super().reset()
        self.__WIN.clear()
        self.__MU_M = 0

    def get_settings(self):
        settings = [str(self.__N) + "." + str(self.__DELTA),
                    "$n$:" + str(self.__N) + ", " +
                    "$\delta$:" + str(self.__DELTA).upper()]
        return settings
