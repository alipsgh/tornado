"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""


class DriftDetectionEvaluator:
    """This class is used to evaluate a drift detection method."""

    @staticmethod
    def calculate_dl_tp_fp_fn(located_points, actual_points, interval):

        actual_drift_points = actual_points.copy()

        num_actual_drifts = len(actual_points)
        num_located_drift_points = len(located_points)

        drift_detection_tp = 0
        drift_detection_dl = []
        for located in located_points:
            for actual in actual_points:
                if actual <= located <= actual + interval:
                    drift_detection_tp += 1
                    drift_detection_dl.append(located - actual - (actual_drift_points.index(actual) + 1))
                    actual_points.remove(actual)
                    break
        drift_detection_dl = sum(drift_detection_dl) + (num_actual_drifts - len(drift_detection_dl)) * interval
        drift_detection_dl /= num_actual_drifts
        drift_detection_fp = num_located_drift_points - drift_detection_tp
        drift_detection_fn = num_actual_drifts - drift_detection_tp

        return drift_detection_dl, drift_detection_tp, drift_detection_fp, drift_detection_fn

