irspack - Implicit recommenders for practitioners
=================================================

Build, compare, and tune recommenders from clicks, views, saves, and purchases.
**irspack** provides a consistent workflow for taking implicit-feedback data
from a practical baseline to an evaluated model candidate.

.. grid:: 1 1 2 2
   :gutter: 2
   :class-container: irspack-quicklinks

   .. grid-item-card:: Prepare your data
      :link: using_your_data
      :link-type: doc

      Convert business IDs and event tables into sparse interaction matrices.

   .. grid-item-card:: Choose a model
      :link: choosing_a_recommender
      :link-type: doc

      Start with a useful baseline and compare models with different biases.

   .. grid-item-card:: Evaluate consistently
      :link: examples/evaluate-recommender
      :link-type: doc

      Compare candidates with the same split, candidate set, and ranking metrics.

   .. grid-item-card:: Tune promising models
      :link: examples/hyperparameter-optimization
      :link-type: doc

      Use Optuna-backed tuning after a model earns further experimentation.

Why irspack?
------------

There is no all-purpose algorithm for implicit-feedback recommendation. The
practical approach is to establish a baseline, compare candidates under one
evaluation protocol, and tune the models that work for your data. irspack keeps
that workflow consistent across algorithms.

Key capabilities include:

   -  `optuna <https://optuna.org/>`_-backed, efficient hyperparameter optimization.
      In particular, `pruning <https://optuna.readthedocs.io/en/stable/tutorial/007_pruning.html?>`_ is used to speed-up the parameter search for several algorithms.
   -  Implementation of several parallelizable algorithms with `nanobind <https://github.com/wjakob/nanobind>`_ and `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_.
      Evaluation of recommenders' performance (which involves score-sorting and ranking metric computation) can be also done efficiently using these technologies.

Install and start
-----------------

Install the published package from PyPI: ::

   pip install irspack

Then begin with :doc:`using_your_data`, or work through the first
:doc:`examples/train-first-recommender` notebook.

Pre-built wheels are published for supported Linux, macOS, and Windows
platforms. To build with CPU-specific compiler optimizations, install from
source with: ::

   CFLAGS="-march=native" pip install git+https://github.com/tohtsky/irspack.git


.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   feature_aware_ials
   api_reference


Reference indices
-----------------

* :ref:`genindex`
* :ref:`search`
