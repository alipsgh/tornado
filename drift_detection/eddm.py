"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Early Drift Detection Method (EDDM) Implementation ***
Paper: Baena-García, Manuel, et al. "Early drift detection method." (2006).
URL: http://www.cs.upc.edu/~abifet/EDDM.pdf
"""

import math

from dictionary.tornado_dictionary import TornadoDic
from drift_detection.detector import SuperDetector


class EDDM(SuperDetector):
    """The Early Drift Detection Method (EDDM) class."""

    DETECTOR_NAME = TornadoDic.EDDM

    def __init__(self):

        super().__init__()

        self.WARNING_LEVEL = 0.95
        self.OUT_CONTROL_LEVEL = 0.9

        self.MINIMUM_NUM_INSTANCES = 30
        self.NUM_INSTANCES_SEEN = 0

        self.MINIMUM_NUM_ERRORS = 30
        self.NUM_ERRORS = 0

        self.P = 0.0  # mean
        self.S_TEMP = 0.0
        self.M2S_max = 0

        self.LATEST_E_LOCATION = 0
        self.SECOND_LATEST_E_LOCATION = 0

    def run(self, prediction_status):
        warning_status = False
        drift_status = False

        self.NUM_INSTANCES_SEEN += 1

        if prediction_status is False:

            self.NUM_ERRORS += 1

            self.SECOND_LATEST_E_LOCATION = self.LATEST_E_LOCATION
            self.LATEST_E_LOCATION = self.NUM_INSTANCES_SEEN
            distance = self.LATEST_E_LOCATION - self.SECOND_LATEST_E_LOCATION

            old_p = self.P
            self.P += (distance - self.P) / self.NUM_ERRORS
            self.S_TEMP += (distance - self.P) * (distance - old_p)

            s = math.sqrt(self.S_TEMP / self.NUM_ERRORS)
            m2s = self.P + 2 * s

            if self.NUM_INSTANCES_SEEN > self.MINIMUM_NUM_INSTANCES:
                if m2s > self.M2S_max:
                    self.M2S_max = m2s
                elif self.NUM_ERRORS > self.MINIMUM_NUM_ERRORS:
                    r = m2s / self.M2S_max
                    if r < self.WARNING_LEVEL:
                        warning_status = True
                    if r < self.OUT_CONTROL_LEVEL:
                        drift_status = True

        return warning_status, drift_status

    def reset(self):
        super().reset()
        self.P = 0.0
        self.S_TEMP = 0.0
        self.NUM_ERRORS = 0
        self.M2S_max = 0

        self.LATEST_E_LOCATION = 0
        self.SECOND_LATEST_E_LOCATION = 0

        self.NUM_INSTANCES_SEEN = 0

    def get_settings(self):
        return [str(self.MINIMUM_NUM_INSTANCES),
                "$n_{min}$:" + str(self.MINIMUM_NUM_INSTANCES)]
