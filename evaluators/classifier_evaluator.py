"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

import math

from dictionary.tornado_dictionary import TornadoDic


class PredictionEvaluator:
    """This class is used to evaluate a classifier."""

    @staticmethod
    def calculate(measure, confusion_matrix, theta=0.000001):
        if measure == TornadoDic.ACCURACY:
            return PredictionEvaluator.calculate_accuracy(confusion_matrix)
        elif measure == TornadoDic.ERROR_RATE:
            return PredictionEvaluator.calculate_error_rate(confusion_matrix)
        elif measure == TornadoDic.PRECISION:
            return PredictionEvaluator.calculate_precision(confusion_matrix, theta)
        elif measure == TornadoDic.RECALL:
            return PredictionEvaluator.calculate_recall(confusion_matrix, theta)
        elif measure == TornadoDic.SPECIFICITY:
            return PredictionEvaluator.calculate_specificity(confusion_matrix, theta)
        elif measure == TornadoDic.F_MEASURE:
            return PredictionEvaluator.calculate_f_measure(confusion_matrix, 1, theta)
        elif measure == TornadoDic.YOUDENS_J:
            return PredictionEvaluator.calculate_youdensj(confusion_matrix, theta)

    @staticmethod
    def print_confusion_matrix(confusion_matrix):
        for k1, v1 in confusion_matrix.items():
            for k2, v2 in confusion_matrix[k1].items():
                print(confusion_matrix[k1][k2], end="\t")
            print()

    @staticmethod
    def calculate_accuracy(confusion_matrix):
        total_sum = 0
        diagonal_sum = 0
        for k1, v1 in confusion_matrix.items():
            for k2, v2 in confusion_matrix[k1].items():
                if k1 == k2:
                    diagonal_sum += confusion_matrix[k1][k2]
                total_sum += confusion_matrix[k1][k2]
        accuracy = diagonal_sum / total_sum
        return accuracy

    @staticmethod
    def calculate_error_rate(confusion_matrix):
        error_rate = 1 - PredictionEvaluator.calculate_accuracy(confusion_matrix)
        return error_rate

    @staticmethod
    def calculate_precision(confusion_matrix, theta):
        precisions = []
        for k1, v1 in confusion_matrix.items():
            true_positive = 0
            false_positive = 0
            for k2, v2 in confusion_matrix[k1].items():
                if k1 == k2:
                    true_positive = confusion_matrix[k1][k2]
                else:
                    false_positive += confusion_matrix[k2][k1]
            precisions.append(true_positive / (true_positive + false_positive + theta))

        precision = 0
        for p in precisions:
            precision += (p / len(precisions))
        return precision

    @staticmethod
    def calculate_recall(confusion_matrix, theta):
        recalls = []
        for k1, v1 in confusion_matrix.items():
            true_positive = 0
            false_negative = 0
            for k2, v2 in confusion_matrix[k1].items():
                if k1 == k2:
                    true_positive = confusion_matrix[k1][k2]
                else:
                    false_negative += confusion_matrix[k1][k2]
            recalls.append(true_positive / (true_positive + false_negative + theta))

        recall = 0
        for p in recalls:
            recall += (p / len(recalls))
        return recall

    @staticmethod
    def calculate_specificity(confusion_matrix, theta):
        true_negatives = []
        false_positives = []
        specificities = []
        for k1, v1 in confusion_matrix.items():
            true_negative = 0
            false_positive = 0
            for k2, v2 in confusion_matrix[k1].items():
                if k1 == k2:
                    for k3, v3 in confusion_matrix.items():
                        for k4, v4 in confusion_matrix[k3].items():
                            if k3 != k1 and k4 != k1:
                                true_negative += confusion_matrix[k3][k4]
                else:
                    false_positive += confusion_matrix[k2][k1]
            true_negatives.append(true_negative)
            false_positives.append(false_positive)
            specificities.append(true_negative / (true_negative + false_positive + theta))
        specificity = 0
        for s in specificities:
            specificity += (s / len(specificities))
        return specificity

    @staticmethod
    def calculate_f_measure(confusion_matrix, beta, theta):
        precision = PredictionEvaluator.calculate_precision(confusion_matrix, theta)
        recall = PredictionEvaluator.calculate_recall(confusion_matrix, theta)
        f_beta = (1 + math.pow(beta, 2)) * ((precision * recall) / (math.pow(beta, 2) * precision + recall + theta))
        return f_beta

    @staticmethod
    def calculate_youdensj(confusion_matrix, beta):
        sensitivity = PredictionEvaluator.calculate_recall(confusion_matrix, beta)
        specificity = PredictionEvaluator.calculate_specificity(confusion_matrix, beta)
        return sensitivity + specificity - 1

    # IMPLEMENT COHEN'S + FLEISS' KAPPA

