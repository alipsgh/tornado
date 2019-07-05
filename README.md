# The Tornado Framework

![Language](https://img.shields.io/badge/language-Python-blue.svg)

Tornado is a framework for data stream mining, written in Python. The framework includes implementations of various incremental/online learning algorithms as well as concept drift detection methods.

You must have Python 3.5 or above (either 32-bit or 64-bit) on your system to run the framework without any error. Note that the **numpy**, **scipy**, **mathplotlib**, and **pympler** packages are used in the Tornado's implementations. You may use the `pip` command in order to install these packages, for example:

`pip install numpy`

Although you can use an installer from https://www.python.org/downloads/ to install Python on your system, I highly recommend **Anaconda**, one of the Python distributions, since it includes the **numpy**, **scipy**, and **mathplotlib** packages by default. You may download one of the Anaconda's installers from https://www.anaconda.com/download/. Please note that, you still need to install the **pympler** package for Anaconda. For that, run the following command in a command prompt or a terminal:

`conda install -c conda-forge pympler`

Once you have all the packages installed, you may run the framework. Three sample codes are prepared to show how you can use the framework. These files are:
* **_github_prequential_test.py_** - This file lets you evaluate an adaptive algorithm, i.e. a pair of a learner and a drift detector, prequentially. In this example, Naive Bayes is the learner and Fast Hoeffding Drift Detection Method (FHDDM) is the detector. You find lists of incremental learners in `tornado/classifier/` and drift detectors in `tornado/drift_detection/`. The outputs in the created project directory are similar to:

<p align="center">
  <img src="/tutorial_img/pr/nb_fhddm.100.png" width="50%"/><br />
  <img src="/tutorial_img/pr/nb_fhddm.100.er.png" width="40%"/>
</p>

* **_github_prequential_multi_test.py_** - This file lets you run multiple adaptive algorithms together against a data stream. While algorithms are learning from instances of a data stream, the framework tells you which adaptive algorithm is optimal by considering _classification_, _adaptation_, and _resource consumption_ measures. The outputs in the created project directory are similar to:

<p align="center">
  <img src="/tutorial_img/multi/sine1_multi_score.png" width="80%"/><br />
  <img src="/tutorial_img/multi/sine1_multi_sine1_cr.png" width="75%"/>
</p>

* **_github_generate_stream.py_** - The file helps you use the Tornado framework for generating synthetic data streams containing concept drifts. You find a list of stream generators in `tornado/streams/generators/`.

### Citation

Please kindly cite the following papers, or thesis, if you plan to use Tornado or any of its components:

1. Pesaranghader, Ali. "__A Reservoir of Adaptive Algorithms for Online Learning from Evolving Data Streams__", Ph.D. Dissertation, Université d'Ottawa/University of Ottawa, 2018. <br />
DOI: http://dx.doi.org/10.20381/ruor-22444
2. Pesaranghader, Ali, et al. "__Reservoir of Diverse Adaptive Learners and Stacking Fast Hoeffding Drift Detection Methods for Evolving Data Streams__", *Machine Learning Journal*, 2018. <br />
Pre-print available at: https://arxiv.org/abs/1709.02457, DOI: https://doi.org/10.1007/s10994-018-5719-z
3. Pesaranghader, Ali, et al. "__A framework for classification in data streams using multi-strategy learning__", *International Conference on Discovery Science*, 2016. <br />
Pre-print available at: http://iwera.ir/~ali/papers/ds2016.pdf, DOI: https://doi.org/10.1007/978-3-319-46307-0_22

<br/>
<br/>

<sub>Ali Pesaranghader © 2019<br/>Under MIT License</sub>
