Get started
===========

Move from an interaction table to a fair offline comparison. If this is your
first time using irspack, follow these guides in order; otherwise, jump directly
to the step that matches your current workflow.

.. grid:: 1 2 2 2
   :gutter: 2
   :class-container: irspack-quicklinks

   .. grid-item-card:: Prepare your data
      :link: using_your_data
      :link-type: doc

      Create stable user and item mappings and choose a split that reflects the
      product scenario.

   .. grid-item-card:: Choose a recommender
      :link: choosing_a_recommender
      :link-type: doc

      Establish a cheap baseline, then compare a few models with different
      assumptions.

   .. grid-item-card:: Train and evaluate
      :link: examples/train-first-recommender
      :link-type: doc

      Work through the first model and evaluate it using ranking metrics.

   .. grid-item-card:: Tune a candidate
      :link: examples/hyperparameter-optimization
      :link-type: doc

      Apply Optuna-backed tuning after the evaluation pipeline is stable.

.. toctree::
   :maxdepth: 1

   using_your_data
   choosing_a_recommender
   examples/train-first-recommender
   examples/evaluate-recommender
   examples/hyperparameter-optimization
   examples/1-vs-100-negative
