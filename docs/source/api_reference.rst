.. _APIReference:

==============
API References
==============

.. currentmodule:: irspack.evaluator

Evaluators
----------
.. autosummary::
    :toctree: api_reference
    :nosignatures:

    Evaluator
    EvaluatorWithColdUser


.. currentmodule:: irspack.recommenders

Recommenders
------------
.. autosummary::
    :toctree: api_reference
    :nosignatures:

    BaseRecommender
    TopPopRecommender
    IALSRecommender
    P3alphaRecommender
    RP3betaRecommender
    TruncatedSVDRecommender
    CosineKNNRecommender
    AsymmetricCosineKNNRecommender
    JaccardKNNRecommender
    TverskyIndexKNNRecommender
    CosineUserKNNRecommender
    AsymmetricCosineUserKNNRecommender
    SLIMRecommender
    DenseSLIMRecommender

A LightFM wrapper for BPR matrix factorization (requires a separate installation of `lightFM <https://github.com/lyst/lightfm>`_).

.. autosummary::
    :toctree: api_reference
    :nosignatures:

    BPRFMRecommender

As a reference code based on neural networks, we have implemented a JAX version of `Mult-VAE <https://arxiv.org/abs/1802.05814>`_,
which requires ``jax``, ``jaxlib``, ``dm-haiku``, and ``optax``:

.. autosummary::
    :toctree: api_reference
    :nosignatures:

    MultVAERecommender


.. currentmodule:: irspack.optimizers

Optimizers
-----------
.. autosummary::
    :toctree: api_reference
    :nosignatures:

    BaseOptimizer
    TopPopOptimizer
    IALSOptimizer
    P3alphaOptimizer
    RP3betaOptimizer
    TruncatedSVDOptimizer
    CosineKNNOptimizer
    AsymmetricCosineKNNOptimizer
    JaccardKNNOptimizer
    TverskyIndexKNNOptimizer
    CosineUserKNNOptimizer
    AsymmetricCosineUserKNNOptimizer
    SLIMOptimizer
    DenseSLIMOptimizer
    MultVAEOptimizer

.. currentmodule:: irspack.split

Split Functions
---------------
.. autosummary::
    :toctree: api_reference
    :nosignatures:

    UserTrainTestInteractionPair
    rowwise_train_test_split
    split_dataframe_partial_user_holdout
    holdout_specific_interactions
