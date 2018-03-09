# The Tornado Framework

Tornado is a framework for data stream mining, written in Python. The framework includes the implementation of various incremental/online learning algorithms as well as concept drift detection methods.

You must have Python 3.5 or above (either 32-bit or 64-bit) on your system to run the framework without any error. Note that the **numpy**, **scipy**, **mathplotlib**, and **pympler** packages are used in the Tornado's implementations. You may use the `pip` command in order to install these packages, for example:

`pip install numpy`

Although you can use an installer from https://www.python.org/downloads/ to install Python on your system, I highly recommend **Anaconda**, one of the Python distributions, since it includes the **numpy**, **scipy**, and **mathplotlib** packages by default. You may download one of the Anaconda's installers from https://www.anaconda.com/download/. Note that, you still need to install the **pympler** package for Anaconda. For that, run the following command in a command prompt or a terminal:

`conda install -c conda-forge pympler`

Once you have all the packages installed, you can run the framework. Three sample codes are prepared to show how you can use the framework. These files are:
* **_github_prequential_test.py_** - This file lets you evaluate an adaptive algorithm, i.e. a pair of a learner and a drift detector, prequentially. In this example, Naive Bayes is the learner and Fast Hoeffding Drift Detection Method (FHDDM) is the detector. You find lists of incremental learners in `tornado/classifier/` and drift detectors in `tornado/drift_detection/`.
* **_github_prequential_multi_test.py_** - This file lets you run multiple adaptive algorithms together against a data stream. While algorithms are learning from instances of a data stream, the framework tells you which adaptive algorithm is optimal by considering _classification_, _adaptation_, and _resource consumption_ measures.
* **_github_generate_stream.py_** - The file helps you use the Tornado framework for generating synthetic data streams containing concept drifts. You find a list of stream generators in `tornado/streams/generators/`.

<br/>
<br/>

<sub>Ali Pesaranghader © 2018<br/>Under MIT License</sub>
