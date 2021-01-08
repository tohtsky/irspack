irspack - Train, Evaluate, and Optimize Recommender Systems with Implicit-Feedback
===================================================================================

**irspack** is a collection of recommender system algorithms for implicit feedback data.

Currently, in my opinion, there is no all-purpose algorithm for the recommendation tasks with implicit-feedback.
So the key is to try out different algorithms, evaluate its performance against validation dataset, and optimize their performance by tuning hyperparameters.
irspack is built to make this procedure easy for you.

Notable features include:

   -  `optuna <https://optuna.org/>`_-backed, efficient hyperparameter optimization.
      In particular, `pruning <https://optuna.readthedocs.io/en/stable/tutorial/007_pruning.html?>`_ is used to speed-up the parameter search for several algorithms.
   -  Implementation of several parallelizable algorithms with `Pybind11 <https://github.com/pybind/pybind11>`_ and `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_.
      Evaluation of recommenders' performance (which involves score-sorting and ranking metric computation) can be also done efficiently using these technologies.

Installation
-------------

If you have a standard Python environment on MacOS/Linux, you can install the library from PyPI using pip: ::

   pip install irspack

The binaries on PyPI have been built to use AVX instruction.
If you want to use AVX2/AVX512 or your environment does not support AVX (e.g. Rosetta2 on Apple Silicon),
install it from source by e.g., : ::

   CFLAGS="-march=native" pip install git+https://github.com/tohtsky/irspack.git


.. toctree::
   :caption: Basic Tutorials
   :maxdepth: 1

   examples/train-first-recommender
   examples/evaluate-recommender
   examples/hyperparameter-optimization

.. toctree::
   :caption: Details
   :maxdepth: 1

   api_reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
