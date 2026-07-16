Choosing a recommender
======================

Recommendation quality depends more on your data and evaluation design than on
the reputation of a particular algorithm. A practical workflow is to establish
a cheap baseline, compare a few models under the same protocol, and tune only
the candidates that earn it.

Where to start
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - Need
     - Start with
     - Why
   * - Pipeline sanity check
     - ``TopPopRecommender``
     - Very fast and easy to interpret. Every personalized model should beat it
       on the metrics that matter to your product.
   * - Strong general baseline
     - ``IALSRecommender``
     - A reliable first choice for implicit feedback and large sparse matrices.
   * - Item-to-item recommendations
     - ``CosineKNNRecommender``
     - Recommendations can be explained through similar previously interacted
       items, and scoring is straightforward.
   * - Sparse interaction data
     - ``RP3betaRecommender``
     - A graph-based model that is often a useful complement to matrix
       factorization.
   * - Sparse linear model
     - ``SLIMRecommender``
     - Learns item-to-item weights directly, at a higher training cost.
   * - Side features and cold start
     - Feature-aware iALS
     - Incorporates user and item features when interaction history is missing
       or limited. This implementation is experimental.

Other included recommenders such as P3alpha, user KNN, EDLAE, NMF, truncated
SVD, and MultVAE are useful additional candidates when their assumptions match
your data or when you need a broader benchmark.

Evaluate the product scenario
-----------------------------

Choose a split before choosing a model. For a feed or marketplace, a temporal
holdout usually resembles deployment better than randomly hiding past events.
A random per-user split remains useful for reproducible benchmarks, but can
leak future behavior into training.

Use the same:

* training and validation interactions;
* candidate item set;
* handling of already-seen items;
* cutoff and metrics;
* user eligibility rules;

for every model in a comparison.

Ranking metrics answer different questions. Recall measures how much relevant
content is retrieved, precision measures how concentrated the list is, and NDCG
rewards relevant items near the top. Coverage, entropy, and the Gini index help
detect a model that obtains accuracy by recommending only a narrow set of
popular items.

A practical sequence
--------------------

1. Run ``TopPopRecommender`` to validate the data and evaluator.
2. Add ``IALSRecommender`` as a strong general-purpose candidate.
3. Add one model with a different bias, such as item KNN or RP3beta.
4. Inspect accuracy, coverage, training time, and recommendation latency.
5. Tune only the most promising candidates on validation data.
6. Evaluate the selected configuration once on an untouched test set.

Offline metrics are filters, not proof of product impact. Before deployment,
also inspect sample recommendations, popularity bias, cold-start behavior, and
failure cases relevant to the business.
