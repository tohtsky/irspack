import json
from typing import List, Tuple, Type

import pandas as pd
from scipy import sparse as sps

from irspack import (
    AsymmetricCosineKNNRecommender,
    BaseRecommender,
    CosineKNNRecommender,
    DenseSLIMRecommender,
    IALSRecommender,
    P3alphaRecommender,
    RP3betaRecommender,
    SLIMRecommender,
    TopPopRecommender,
    TverskyIndexKNNRecommender,
)
from irspack.dataset.movielens import MovieLens1MDataManager
from irspack.evaluation import EvaluatorWithColdUser
from irspack.split import split_dataframe_partial_user_holdout

if __name__ == "__main__":

    BASE_CUTOFF = 20

    data_manager = MovieLens1MDataManager()
    df_all = data_manager.read_interaction()

    data_all, _ = split_dataframe_partial_user_holdout(
        df_all,
        "userId",
        "movieId",
        test_user_ratio=0.2,
        val_user_ratio=0.2,
        heldout_ratio_test=0.5,
        heldout_ratio_val=0.5,
    )

    data_train = data_all["train"]
    data_val = data_all["val"]
    data_test = data_all["test"]

    X_train_val_all: sps.csr_matrix = sps.vstack(
        [data_train.X_all, data_val.X_all], format="csr"
    )
    valid_evaluator = EvaluatorWithColdUser(
        input_interaction=data_val.X_train,
        ground_truth=data_val.X_test,
        cutoff=BASE_CUTOFF,
    )
    test_evaluator = EvaluatorWithColdUser(
        input_interaction=data_test.X_train,
        ground_truth=data_test.X_test,
        cutoff=BASE_CUTOFF,
    )

    test_results = []
    validation_results = []

    test_configs: List[Tuple[Type[BaseRecommender], int]] = [
        (TopPopRecommender, 1),
        (CosineKNNRecommender, 40),
        (AsymmetricCosineKNNRecommender, 40),
        (TverskyIndexKNNRecommender, 40),
        (P3alphaRecommender, 40),
        (RP3betaRecommender, 40),
        (IALSRecommender, 40),
        (DenseSLIMRecommender, 20),
        (SLIMRecommender, 40),  # time consuming
    ]
    for recommender_class, n_trials in test_configs:
        recommender_name = recommender_class.__name__
        (best_param, validation_result_df) = recommender_class.tune(
            data_train.X_all, valid_evaluator, n_trials=n_trials, random_seed=0
        )
        validation_result_df["recommender_name"] = recommender_name
        validation_results.append(validation_result_df)
        pd.concat(validation_results).to_csv(f"validation_scores.csv")
        test_recommender = recommender_class(X_train_val_all, **best_param)
        test_recommender.learn()
        test_scores = test_evaluator.get_scores(test_recommender, [5, 10, 20])

        test_results.append(
            dict(name=recommender_name, best_param=best_param, **test_scores)
        )
        with open("test_results.json", "w") as ofs:
            json.dump(test_results, ofs, indent=2)
