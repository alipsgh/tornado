import math


class Transition:

    @staticmethod
    def sigmoid(x, win):
        p = -(4 / win) * (x - (win / 2))
        y = 1 / (1 + math.pow(math.e, p))
        return y
