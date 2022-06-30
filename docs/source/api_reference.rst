.. _APIReference:

==============
API References
==============

.. currentmodule:: irspack.evaluation

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
    get_recommender_class

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
    split_last_n_interaction_df

.. currentmodule:: irspack.utils

Utilities
---------

.. autosummary::
    :toctree: api_reference
    :nosignatures:

    ItemIDMapper
    IDMapper

.. currentmodule:: irspack.dataset

Dataset
-------

.. autosummary::
    :toctree: api_reference
    :nosignatures:

    MovieLens1MDataManager
    MovieLens100KDataManager
    MovieLens20MDataManager
    NeuMFML1MDownloader
    NeuMFPinterestDownloader
