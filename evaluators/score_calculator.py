"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""


from pympler import asizeof

from dictionary.tornado_dictionary import TornadoDic
from evaluators.classifier_evaluator import PredictionEvaluator


class LearnersScoreCalculator:
    """This class is used to calculate scores of (classifier, detector) pairs."""

    @staticmethod
    def calculate_emr(learners, error_weight=1, memory_weight=1, runtime_weight=1, lb=1, ub=10):

        learners_names = []
        learners_errors = []
        learners_runtime = []
        learners_memory_usages = []
        learners_emr_scores = []

        for learner in learners:
            learners_names.append(learner.LEARNER_NAME.title())
            learners_errors.append(PredictionEvaluator.calculate(TornadoDic.ERROR_RATE, learner.get_confusion_matrix()))
            learners_runtime.append(learner.get_total_running_time())
            if memory_weight != -1:
                learners_memory_usages.append(asizeof.asizeof(learner))
            else:
                learners_memory_usages.append(0)

        if len(learners) != 1:
            err_min, err_max = LearnersScoreCalculator.get_min_max(learners_errors)
            run_min, run_max = LearnersScoreCalculator.get_min_max(learners_runtime)
            mem_min, mem_max = LearnersScoreCalculator.get_min_max(learners_memory_usages)

            for i in range(0, len(learners_names)):
                error = LearnersScoreCalculator.scale(learners_errors[i], err_min, err_max, lb, ub)
                runtime = LearnersScoreCalculator.scale(learners_runtime[i], run_min, run_max, lb, ub)
                memory_usage = LearnersScoreCalculator.scale(learners_memory_usages[i], mem_min, mem_max, lb, ub)
                learners_emr_scores.append(LearnersScoreCalculator.__cal_emr(ub, error, runtime, memory_usage, error_weight, runtime_weight, memory_weight))

            return learners_emr_scores, learners_errors, learners_memory_usages, learners_runtime
        else:
            return None, learners_errors, learners_memory_usages, learners_runtime

    @staticmethod
    def __cal_emr(ub, error, runtime, memory_usage, ew, rw, mw):
        emr_score = ub - (error * ew + runtime * rw + memory_usage * mw) / (ew + rw + mw)
        return emr_score

    @staticmethod
    def get_min_max(elements):
        minimum = min(elements)
        maximum = max(elements)
        return minimum, maximum

    @staticmethod
    def scale(x, minimum, maximum, lower_bound, upper_bound):
        if maximum != minimum:
            scaled_x = lower_bound + ((x - minimum) * (upper_bound - lower_bound)) / (maximum - minimum)
        else:
            scaled_x = 0
        return scaled_x