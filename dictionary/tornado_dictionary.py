"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""


class TornadoDic:
    """This class is contains the terminologies in the Tornado framework."""

    EMPTY = "EMPTY"
    NUMERIC_ATTRIBUTE = "NUMERIC"
    NOMINAL_ATTRIBUTE = "NOMINAL"

    SUM = "SUM"
    SUM_SQUARE = "SUM_SQUARE"

    NAIVE_BAYES = "NAIVE BAYES"
    NB = "NB"
    DECISION_STUMP = "DECISION STUMP"
    DS = "DS"
    HOEFFDING_TREE = "HOEFFDING TREE"
    HT = "HT"
    PERCEPTRON = "PERCEPTRON"
    PR = "PR"
    NEURAL_NETWORK = "NEURAL NETWORK"
    NN = "NN"
    K_NN = "K NEAREST NEIGHBOURS"
    KNN = "KNN"

    MDDM_A = "MDDM.A"
    MDDM_G = "MDDM.G"
    MDDM_E = "MDDM.E"
    FHDDMS = "FHDDMS"
    FHDDMS_add = "FHDDMS.add"
    FHDDM = "FHDDM"
    HDDM_A_test = "HDDM.A.test"
    HDDM_W_test = "HDDM.W.test"
    SeqDrift2 = "SeqDrift2"
    EWMA = "EWMA"
    ADWIN = "ADWIN"
    DDM = "DDM"
    EDDM = "EDDM"
    RDDM = "RDDM"
    PH = "PageHinkley"
    CUSUM = "CUSUM"
    NO_DETECTION = "NO_DETECTION"

    MAJORITY_COUNTING = "MAJORITY COUNTING"
    MC = "MC"

    LOADABLE = "LOADABLE"
    TRAINABLE = "TRAINABLE"
    NUM_CLASSIFIER = "CLASSIFIER FOR NUMERIC ATTRIBUTES"
    NOM_CLASSIFIER = "CLASSIFIER FOR NOMINAL ATTRIBUTES"

    RUNNING_TIME = "RUNNING TIME"
    TRAINING_TIME = "TRAINING TIME"
    TESTING_TIME = "TESTING TIME"

    ACCURACY = "ACCURACY"
    ERROR_RATE = "ERROR RATE"
    PRECISION = "PRECISION"
    RECALL = "RECALL"
    SPECIFICITY = "SPECIFICITY"
    F_MEASURE = "F MEASURE"
    YOUDENS_J = "YOUDEN'S J"

    GLOBAL_PERFORMANCE = "GLOBAL PERFORMANCE"
    PERIODICAL_PERFORMANCE = "PERIODICAL PERFORMANCE"

    @staticmethod
    def get_short_names(name):
        if name == TornadoDic.NAIVE_BAYES:
            name = TornadoDic.NB
        elif name == TornadoDic.DECISION_STUMP:
            name = TornadoDic.DS
        elif name == TornadoDic.HOEFFDING_TREE:
            name = TornadoDic.HT
        elif name == TornadoDic.PERCEPTRON:
            name = TornadoDic.PR
        elif name == TornadoDic.NEURAL_NETWORK:
            name = TornadoDic.NN
        elif name.endswith('NEAREST NEIGHBORS'):
            name = name.replace('NEAREST NEIGHBORS', 'NN')
        return name
