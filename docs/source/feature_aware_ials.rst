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

Feature-only cold-start embeddings can then be computed from features.  These
methods solve the iALS least-squares system with an empty interaction history,
using the same solver type selected for training.  Consequently, the result
includes the loss on unobserved interactions and is generally not exactly
``A f`` or ``B g``:

.. code-block:: python

   new_user_features = ...  # shape: (n_new_users, n_user_features)
   new_user_embedding = rec.compute_user_embedding_from_features(
       new_user_features
   )
   scores = rec.get_score_cold_user_from_features(new_user_features)

   new_item_features = ...  # shape: (n_new_items, n_item_features)
   new_item_embedding = rec.compute_item_embedding_from_features(
       new_item_features
   )
   item_scores = rec.get_score_from_item_features(
       user_indices=[0, 1, 2],
       item_features=new_item_features,
   )

If both interaction history and side features are available at prediction time,
pass the feature matrix to the normal transform API.  This solves the same
regularized least-squares problem used during training, instead of returning
only ``A f`` or ``B g``.

.. code-block:: python

   X_new_user = ...  # shape: (n_new_users, n_items)
   new_user_features = ...  # shape: (n_new_users, n_user_features)

   hybrid_user_embedding = rec.compute_user_embedding(
       X_new_user,
       user_features=new_user_features,
   )
   hybrid_scores = rec.get_score_cold_user(
       X_new_user,
       user_features=new_user_features,
   )

   X_new_items = ...  # shape: (n_users, n_new_items)
   new_item_features = ...  # shape: (n_new_items, n_item_features)

   hybrid_item_embedding = rec.compute_item_embedding(
       X_new_items,
       item_features=new_item_features,
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
