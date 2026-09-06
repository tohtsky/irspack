Feature-aware iALS
==================

``IALSRecommender`` supports an experimental feature-aware extension of
implicit ALS (iALS). It is useful when user or item side information is available
and recommendations must also work for cold-start users or items.

Here, "feature-aware iALS" refers to irspack's implementation of an existing
model family.  Its item-feature formulation follows content-aware WMF
[Liang2015]_, which learns a linear map from fixed content features to an
item-factor prior mean; feature-centered priors also appear in RLFM
[Agarwal2009]_.  irspack supports features for either side, retains iALS's
frequency-aware regularization [Rendle2022]_, and uses a different
empty-history inference policy from [Liang2015]_.

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

where ``f_u`` and ``g_i`` are user and item feature vectors, and ``A`` and ``B`` are linear maps from feature space to the latent embedding space.
In other words, the learned user/item embeddings are residual embeddings around a feature-predicted prior.

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
feature-predicted embeddings.

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

Relative to ordinary iALS, only the right-hand side gains ``r_u A f_u``, so
the same Cholesky and conjugate-gradient solvers can be used.  [Liang2015]_
also alternates factor and linear feature-map updates.

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

[Liang2015]_ instead uses the prior mean directly for new items.  Empty-history
ALS also fits unobserved pairs as zero targets, so it can shrink or rotate that
embedding; this documentation does not compare the two policies.

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

Random interaction holdouts may leave most evaluated items warm and thus
underrepresent newly arriving items.  For a catalog with arriving items,
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

This experiment asks whether feature-aware iALS improves recommendation over a
production-like temporal catalog.  It evaluates the combined capability of
regularizing embeddings for known items and scoring items unseen during
training; it does not attempt to isolate the contribution of each mechanism.

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

Feature-aware iALS achieved 1.90 times the NDCG and approximately 5 times the
catalog coverage of ordinary iALS in this run.

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

Feature-aware iALS also achieves 5.26 times the NDCG of TopPop.  Recall and hit
rate are approximately doubled relative to ordinary iALS.

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

In this run, higher accuracy accompanies broader aggregate recommendation
coverage.  Feature-aware iALS covers 53.6% of the period catalog, compared
with 10.8% for ordinary iALS and 0.5% for TopPop, and also improves both Gini
index and entropy.  These metrics describe aggregate exposure concentration;
they do not establish the cause of the accuracy gain or measure alignment
with individual users' preferences for less popular articles.

This comparison includes ordinary iALS and TopPop, but not other feature-based
methods such as post-hoc feature-to-factor regression.  It therefore does not
isolate the benefit of joint learning or establish an improvement over prior
content-aware WMF methods.

The period-level exposure catalog is still an approximation.  It uses the
union of impressions from the evaluation period for every user, whereas a
strict logged-policy evaluation would rank only the candidates in each
individual impression.  Results should also be checked across additional
temporal splits, random seeds, and content representations before drawing a
general conclusion about feature-aware iALS.

References
----------

.. [Liang2015] Dawen Liang, Minshu Zhan, and Daniel P. W. Ellis.
   `Content-Aware Collaborative Music Recommendation Using Pre-trained Neural Networks
   <https://ismir2015.uma.es/articles/290_Paper.pdf>`_.
   ISMIR 2015, pp. 295–301.  Section 3.2 describes the linear prior map and
   alternating updates.  The authors also published
   `content_wmf <https://github.com/dawenl/content_wmf>`_.

.. [Agarwal2009] Deepak Agarwal and Bee-Chung Chen.
   `Regression-based latent factor models
   <https://doi.org/10.1145/1557019.1557029>`_.
   KDD 2009, pp. 19–28.

.. [Rendle2022] Steffen Rendle, Walid Krichene, Li Zhang, and Yehuda Koren.
   `Revisiting the Performance of iALS on Item Recommendation Benchmarks
   <https://arxiv.org/abs/2110.14037>`_.
   RecSys 2022, pp. 427–435.
