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
    DenseSLIMRecommender
    P3alphaRecommender
    RP3betaRecommender
    TruncatedSVDRecommender
    CosineKNNRecommender
    AsymmetricCosineKNNRecommender
    JaccardKNNRecommender
    TverskyIndexKNNRecommender

A LightFM wrapper for BPR matrix factorization (requires a separate installation of `lightFM <https://github.com/lyst/lightfm>`_).

.. autosummary::
    :toctree: api_reference
    :nosignatures:

    BPRFMRecommender



.. currentmodule:: irspack.optimizers

Optimizers
-----------
.. autosummary::
    :toctree: api_reference
    :nosignatures:

    BaseOptimizer
    TopPopOptimizer
    IALSOptimizer
    DenseSLIMOptimizer
    P3alphaOptimizer
    RP3betaOptimizer
    TruncatedSVDOptimizer
    CosineKNNOptimizer
    AsymmetricCosineKNNOptimizer
    JaccardKNNOptimizer
    TverskyIndexKNNOptimizer
