"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""


import numpy as np
from scipy.stats.mstats import rankdata


class ScoreProcessor:
    """This class is used for processing scores of (classifier, detector) pairs."""

    @staticmethod
    def normalize_matrix(matrix, theta=0.0000001):
        min_col_elements = np.min(matrix, axis=0)
        max_col_elements = np.max(matrix, axis=0)
        rng_col_elements = max_col_elements - min_col_elements
        nrm = np.nan_to_num((1 - (max_col_elements - matrix) / (rng_col_elements + theta))).tolist()

        return nrm

    @staticmethod
    def calculate_weighted_scores(matrix, weights):
        scores = (1 - np.sum(np.multiply(matrix, weights), axis=1) / np.sum(weights)).tolist()
        return scores

    @staticmethod
    def multiply_class_adapt_scores(class_scores, adapt_scores):
        return (np.multiply(class_scores, adapt_scores)).tolist()

    @staticmethod
    def penalize_high_dfp(fp_level, fp_index, p_ratio, matrix):
        max_col_elements = (np.max(matrix, axis=0) * p_ratio).tolist()
        for i in range(0, len(matrix)):
            if matrix[i][fp_index] > fp_level:
                matrix[i] = max_col_elements
        return matrix

    @staticmethod
    def rank_matrix(current_stats):
        ranked_stats = rankdata(current_stats, axis=0)
        return ranked_stats
