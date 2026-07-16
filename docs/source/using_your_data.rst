Using your own interaction data
===============================

Most irspack workflows begin with an event table containing at least a user ID
and an item ID. Clicks, views, saves, plays, and purchases are all examples of
implicit feedback: they indicate behavior, not necessarily preference.

Prepare the event table
-----------------------

Start with one row per event. Remove events that should not count as positive
feedback, and decide whether repeated events should increase interaction
strength or be collapsed into a binary signal.

.. code-block:: python

   import pandas as pd
   from irspack import df_to_sparse

   events = pd.DataFrame(
       {
           "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
           "item_id": ["A", "B", "B", "C", "A", "D"],
           "timestamp": pd.to_datetime(
               ["2025-01-01", "2025-01-04", "2025-01-02",
                "2025-01-05", "2025-01-03", "2025-01-06"]
           ),
       }
   )

   X, user_ids, item_ids = df_to_sparse(events, "user_id", "item_id")

``user_ids[i]`` is the business ID represented by row ``i`` of ``X``;
``item_ids[j]`` is the ID represented by column ``j``. Persist these arrays with
the trained model. A model score is indexed by matrix column, not by the
original item ID.

If the event table has a numeric strength column, pass it as ``rating_column``.
Be deliberate about weighting: raw event counts can allow a small number of
highly active users to dominate training. Binary values are a sensible first
baseline.

Create a leakage-aware split
----------------------------

When timestamps are available, hold out the latest events for each user before
creating sparse matrices. Supply the training ID arrays when converting the
validation frame so both matrices have exactly the same shape and mapping.

.. code-block:: python

   from irspack import df_to_sparse
   from irspack.split import split_last_n_interaction_df

   train_events, validation_events = split_last_n_interaction_df(
       events,
       user_column="user_id",
       timestamp_column="timestamp",
       n_heldout=1,
   )

   X_train, user_ids, item_ids = df_to_sparse(
       train_events, "user_id", "item_id"
   )
   X_validation, _, _ = df_to_sparse(
       validation_events,
       "user_id",
       "item_id",
       user_ids=user_ids,
       item_ids=item_ids,
   )

This intentionally excludes validation users or items that never appeared in
training. Evaluate those separately as cold-start cases rather than silently
mixing them into a warm-start metric.

Train and compare baselines
---------------------------

.. code-block:: python

   from irspack import Evaluator, IALSRecommender, TopPopRecommender

   evaluator = Evaluator(X_validation, cutoff=3)

   for recommender_class in (TopPopRecommender, IALSRecommender):
       model = recommender_class(X_train).learn()
       metrics = evaluator.get_score(model)
       print(recommender_class.__name__, metrics["ndcg"])

Start with TopPop to catch mapping or evaluation mistakes. Then compare iALS
and at least one model with a different inductive bias. Keep an untouched test
period for the final configuration; repeatedly consulting it turns it into
another validation set.

Map recommendations back to item IDs
------------------------------------

.. code-block:: python

   import numpy as np

   user_index = np.flatnonzero(user_ids == "u1")[0]
   scores = model.get_score_remove_seen(np.array([user_index]))[0]
   top_columns = np.argsort(scores)[::-1][:10]
   recommended_item_ids = item_ids[top_columns]

``get_score_remove_seen`` assigns negative infinity to items already present in
the training row. Apply any product constraints—availability, region, safety,
or inventory—before selecting the final ranked list.

Operational checks
------------------

Before treating an offline result as production-ready, check:

* the fraction of users and items absent from training;
* popularity and catalog coverage of recommendations;
* memory use for the interaction matrix and model;
* batch scoring throughput at the intended cutoff;
* representative recommendation examples and obvious failure modes;
* whether the evaluation time window matches retraining frequency.

See :doc:`choosing_a_recommender` for candidate models and evaluation trade-offs.
