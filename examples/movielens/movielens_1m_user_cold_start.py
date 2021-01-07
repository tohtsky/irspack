"""
An example for user cold-start scenario.
CB2CF Requires the following additional dependencies to run.

 - lightfm
 - jax
 - jaxlib
 - dm-haiku
 - optax

"""
import json
import os
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn.model_selection import train_test_split

from irspack.dataset.movielens import MovieLens1MDataManager
from irspack.definitions import UserIndexArray
from irspack.user_cold_start import (
    BaseUserColdStartOptimizer,
    CB2BPRFMOptimizer,
    CB2IALSOptimizer,
    CB2TruncatedSVDOptimizer,
    LinearMethodOptimizer,
    TopPopularOptimizer,
    UserCBCosineKNNOptimizer,
    UserColdStartEvaluator,
)
from irspack.utils.encoders import (
    BinningEncoder,
    CategoricalValueEncoder,
    DataFrameEncoder,
)

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["IRSPACK_NUM_THREADS_DEFAULT"] = "8"


if __name__ == "__main__":
    BASE_CUTOFF = 20
    loader = MovieLens1MDataManager()
    user_df = loader.read_user_info()
    ratings = loader.read_interaction()
    user_ids_unique: List[int] = np.unique(ratings.userId)
    movie_ids_unique: List[int] = np.unique(ratings.movieId)
    movie_id_to_index = {id_: i for i, id_ in enumerate(movie_ids_unique)}

    def df_to_sparse(df: pd.DataFrame) -> Tuple[UserIndexArray, sps.csr_matrix]:
        unique_uids, row = np.unique(df.userId, return_inverse=True)
        col = df.movieId.map(movie_id_to_index)
        return (
            unique_uids,
            sps.csr_matrix(
                (np.ones(df.shape[0]), (row, col)),
                shape=(len(unique_uids), len(movie_id_to_index)),
            ),
        )

    train_uids, test_uids = train_test_split(
        user_ids_unique, test_size=0.2, random_state=42
    )
    train_uids, X_train = df_to_sparse(ratings[ratings.userId.isin(train_uids)])
    test_uids, X_test = df_to_sparse(ratings[ratings.userId.isin(test_uids)])

    ### Preprocess user data
    user_df["zip_first"] = user_df.zipcode.str[0]
    columns = ["occupation", "zip_first"]
    encoder_all = DataFrameEncoder()
    encoder_all.add_column(
        "gender", CategoricalValueEncoder(user_df["gender"])
    ).add_column("age", BinningEncoder(user_df["age"], n_percentiles=10)).add_column(
        "occupation", CategoricalValueEncoder(user_df["occupation"])
    ).add_column(
        "zip_first", CategoricalValueEncoder(user_df["zip_first"])
    )

    user_info_train = encoder_all.transform_sparse(user_df.reindex(train_uids))
    user_info_test = encoder_all.transform_sparse(user_df.reindex(test_uids))

    test_evaluator = UserColdStartEvaluator(X_test, user_info_test)

    trial_configurations: List[Tuple[Type[BaseUserColdStartOptimizer], int]] = [
        (TopPopularOptimizer, 1),
        (UserCBCosineKNNOptimizer, 20),
        (LinearMethodOptimizer, 10),
    ]
    test_results: List[Dict[str, Any]] = []
    for optimizer_class, n_trials in trial_configurations:
        best_param = optimizer_class.split_and_optimize(
            X_train, user_info_train, n_trials=n_trials
        )
        rec = optimizer_class.recommender_class(X_train, user_info_train, **best_param)
        rec.learn()
        test_results.append(
            dict(
                recommender_name=optimizer_class.recommender_class.__name__,
                test_result=test_evaluator.get_score(rec),
            )
        )
    for cboptim_class in [
        CB2TruncatedSVDOptimizer,
        CB2IALSOptimizer,
        CB2BPRFMOptimizer,
    ]:
        rec, best_config, best_nn_config = cboptim_class.split_and_optimize(
            X_train, user_info_train, n_trials=20
        )
        test_results.append(
            dict(
                recommender_name=cboptim_class.recommender_class.__name__,
                test_result=test_evaluator.get_score(rec),
            )
        )
    with open("test_results.json", "w") as ofs:
        json.dump(test_results, ofs, indent=2)
