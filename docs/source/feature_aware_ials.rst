Feature-aware iALS
==================

``IALSRecommender`` supports an experimental feature-aware extension of
implicit ALS.  It is useful when user or item side information is available
and recommendations must also work for cold-start users or items.

The standard iALS model predicts a score by the inner product of user and item
embeddings:

.. math::

   \hat r_{ui} = x_u^\top y_i.

Feature-aware iALS keeps the same scoring form, but regularizes each embedding
toward an embedding predicted from side features:

.. math::

   x_u \sim \mathcal{N}(A f_u, r_u^{-1} I),
   \qquad
   y_i \sim \mathcal{N}(B g_i, r_i^{-1} I),

where ``f_u`` and ``g_i`` are user and item feature vectors, and ``A`` and
``B`` are linear maps from feature space to the latent embedding space.  In
other words, the learned user/item embeddings are residual embeddings around a
feature-predicted prior.

Objective
---------

With the existing iALS loss and frequency-aware regularization, the optimized
objective is:

.. math::

   \sum_{u,i} c_{ui} (p_{ui} - x_u^\top y_i)^2
   + \sum_u r_u \lVert x_u - A f_u \rVert^2
   + \sum_i r_i \lVert y_i - B g_i \rVert^2
   + \lambda_A \lVert A \rVert_F^2
   + \lambda_B \lVert B \rVert_F^2.

The frequency-aware terms ``r_u`` and ``r_i`` are the same regularization
strengths used by normal iALS, but their centers are replaced by the
feature-predicted embeddings.  Thus the side-feature model does not introduce
a second embedding regularization coefficient.

Optimization
------------

The implementation uses block coordinate descent:

1. Update user embeddings ``X`` by the existing iALS linear solver with an
   additional feature-prior term.
2. Update the user feature map ``A`` by multi-output ridge regression.
3. Update item embeddings ``Y`` symmetrically.
4. Update the item feature map ``B`` by multi-output ridge regression.

For example, the user embedding update solves:

.. math::

   \left(
       \sum_i c_{ui} y_i y_i^\top + r_u I
   \right) x_u
   =
   \sum_i c_{ui} p_{ui} y_i + r_u A f_u.

This is the usual iALS system with one extra diagonal term and one extra right
hand-side term, so the same Cholesky and conjugate-gradient solvers can be used.

The feature map update is a ridge regression:

.. math::

   A^\top =
   (F^\top R F + \lambda_A I)^{-1} F^\top R X,

where ``R`` is diagonal with entries ``r_u``.

Usage
-----

Pass user and/or item feature matrices to ``IALSRecommender``.  Features can be
either sparse matrices such as one-hot category features, or dense
``numpy.ndarray`` matrices such as text or image embeddings.  Feature rows must
align with the rows and columns of the training interaction matrix.

.. code-block:: python

   from irspack import IALSRecommender

   X = ...  # shape: (n_users, n_items)
   user_features = ...  # sparse or dense; shape: (n_users, n_user_features)
   item_features = ...  # sparse or dense; shape: (n_items, n_item_features)

   rec = IALSRecommender(
       X,
       user_features=user_features,
       item_features=item_features,
       lambda_user_feature=1e-3,
       lambda_item_feature=1e-3,
       solver_type="CG",
       loss_type="ORIGINAL",
   ).learn()

Evaluation with new items
-------------------------

Supply user history over training items, ground truth ordered as training items
followed by new items, and the features of those new items:

.. code-block:: python

   from irspack import EvaluatorWithColdUser

   X_history = ...  # shape: (n_eval_users, n_training_items)
   X_target = ...   # shape: (n_eval_users, n_training_items + n_new_items)

   evaluator = EvaluatorWithColdUser(
       X_history,
       X_target,
       cold_item_features=new_item_features,
       cutoff=20,
   )
   result = evaluator.get_scores(rec, [20])

``EvaluatorWithColdUser`` prepares new-item embeddings once and reuses them
while scoring user minibatches.  The result includes ``catalog_coverage@20``.
Recommenders without feature-only item support remain valid baselines: their
new-item scores are unavailable and are not included in recommendation counts.

For scoring without an evaluator,
``get_score_cold_user_with_item_features(X_history, new_item_features)``
returns training-item columns followed by new-item columns in feature row
order.

Lower-level transforms
----------------------

Feature-only embeddings solve the iALS least-squares system with empty
interaction history, so they include the loss on unobserved interactions and
are generally not exactly ``A f`` or ``B g``:

.. code-block:: python

   new_user_embedding = rec.compute_user_embedding_from_features(new_user_features)
   new_item_embedding = rec.compute_item_embedding_from_features(new_item_features)

When both interaction history and features are available, pass both to the
normal transform API:

.. code-block:: python

   user_embedding = rec.compute_user_embedding(
       X_new_user, user_features=new_user_features
   )
   item_embedding = rec.compute_item_embedding(
       X_new_items, item_features=new_item_features
   )

Important parameters
--------------------

``lambda_user_feature`` and ``lambda_item_feature``
   Ridge regularization strengths for the user and item feature maps.

``feature_warmup_epochs``
   Number of initial epochs trained as ordinary iALS before enabling the
   feature-aware updates.

Limitations
-----------

- ``solver_type="IALSPP"`` is not supported with feature-aware iALS.
- Feature map updates currently form ``F.T @ F`` explicitly and solve it by
  Cholesky decomposition.  Extremely high-dimensional feature matrices may
  require additional dimensionality reduction or a future iterative ridge
  solver.
- The feature-only cold-start APIs use an empty interaction history together
  with the side-feature prior.  The hybrid
  ``compute_user_embedding(..., user_features=...)`` and
  ``compute_item_embedding(..., item_features=...)`` methods additionally
  account for observed interactions.

Evaluation guidance
-------------------

Random interaction holdouts often understate the benefit of item features:
nearly every evaluated item is already warm and ordinary collaborative
filtering can learn its embedding directly.  For a production-like comparison,
split all interactions at global time boundaries, keep post-cutoff items out of
the training interaction matrix, and fit every feature transformer on
training-period data only.  Define the recommendation catalog from items that
were eligible for exposure during each evaluation period.  Report accuracy on
all targets together with item diversity or coverage.

MIND-small temporal example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``MINDDataManager`` provides a compact public dataset suited to this setup.
MIND-small has timestamped click impressions and article category, title,
abstract, linked-entity, and knowledge-graph features.  Its official training
and development archives are chronologically separated.  Note that the
``history`` field does not contain timestamps for individual clicks;
``read_interaction()`` therefore uses only timestamped positive impressions so
that it does not invent event times or leak future information.

The complete experiment is in
``examples/mind/mind_small_feature_aware_ials.py``.  Run it with:

.. code-block:: console

   uv run python examples/mind/mind_small_feature_aware_ials.py \
       --n-trials 100 \
       --output examples/mind/mind_feature_aware_ials_test.json

The experiment uses the following evaluation design:

- The final calendar day of the official train archive is validation.  The
  chronologically later official dev archive is test.
- The recommendation catalog contains only articles exposed in at least one
  impression during the corresponding evaluation period.  Pre-cutoff items
  outside this catalog remain available for user-history and embedding lookup,
  but cannot be recommended.
- The training interaction matrix contains only items clicked before the
  cutoff.  Post-cutoff items are not inserted as zero-interaction columns.
- TF-IDF, truncated SVD, and category encoders are fitted only on pre-cutoff
  items.  The feature-aware model creates embeddings for post-cutoff items
  only when it is evaluated.
- Only users with pre-cutoff history are included.  Fully cold users and their
  fallback policy are intentionally a separate problem.
- Ordinary and feature-aware iALS are tuned independently with
  ``IALSRecommender.tune``.  Both search ``n_components``, ``alpha0``, and
  ``reg``; feature-aware iALS also searches ``lambda_item_feature``.  Early
  stopping selects the final number of training epochs.

Validation and test both evaluate all target clicks together and report
accuracy, diversity, and catalog coverage.  Test also includes TopPop as an
independent baseline.

The example passes warm-item history, expanded ground truth, and new-item
features directly to ``EvaluatorWithColdUser``.  The same evaluator is used for
early stopping, final feature-aware evaluation, and ordinary iALS or TopPop
baselines; no model-specific evaluation adapter is required.

Example result
^^^^^^^^^^^^^^

The following result was obtained with 100 tuning trials per model.  These are
test-period metrics at cutoff 20; they are an example from one dataset and
temporal split, not a general performance guarantee.

.. list-table:: Accuracy on all test targets
   :header-rows: 1
   :widths: 28 24 24 24

   * - Metric
     - TopPop
     - iALS
     - Feature-aware iALS
   * - NDCG@20
     - 0.00187
     - 0.00516
     - 0.00982
   * - Recall@20
     - 0.00511
     - 0.01127
     - 0.02221
   * - Hit@20
     - 0.01010
     - 0.02288
     - 0.04510

Feature-aware iALS achieves 1.90 times the NDCG of ordinary iALS and 5.26
times that of TopPop on all targets.  Recall and hit rate are also approximately
doubled relative to ordinary iALS.

.. list-table:: Recommendation diversity on all test targets
   :header-rows: 1
   :widths: 28 24 24 24

   * - Metric
     - TopPop
     - iALS
     - Feature-aware iALS
   * - Gini index@20 (lower is better)
     - 0.9982
     - 0.9783
     - 0.9305
   * - Entropy@20 (higher is better)
     - 3.071
     - 5.776
     - 6.897
   * - Catalog coverage@20
     - 0.0050
     - 0.1084
     - 0.5364

In this run, the accuracy gain does not come from concentrating recommendations
on a smaller set of popular articles.  Feature-aware iALS covers 53.6% of the
period catalog, compared with 10.8% for ordinary iALS and 0.5% for TopPop, and
also improves both Gini index and entropy.

The period-level exposure catalog is still an approximation.  It uses the
union of impressions from the evaluation period for every user, whereas a
strict logged-policy evaluation would rank only the candidates in each
individual impression.  Results should also be checked across additional
temporal splits, random seeds, and content representations before drawing a
general conclusion about feature-aware iALS.
