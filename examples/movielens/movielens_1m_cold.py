import json
import logging
import os
from typing import List, Tuple, Type

import pandas as pd
from scipy import sparse as sps

from irspack.dataset.movielens import MovieLens1MDataManager
from irspack.evaluator import EvaluatorWithColdUser
from irspack.optimizers import (  # SLIMOptimizer,; MultVAEOptimizer,
    AsymmetricCosineKNNOptimizer,
    BaseOptimizer,
    CosineKNNOptimizer,
    DenseSLIMOptimizer,
    IALSOptimizer,
    P3alphaOptimizer,
    RP3betaOptimizer,
    TopPopOptimizer,
    TverskyIndexKNNOptimizer,
)
from irspack.split import split_dataframe_partial_user_holdout

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["IRSPACK_NUM_THREADS_DEFAULT"] = "8"

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

    test_configs: List[Tuple[Type[BaseOptimizer], int]] = [
        (TopPopOptimizer, 1),
        (CosineKNNOptimizer, 40),
        (AsymmetricCosineKNNOptimizer, 40),
        (TverskyIndexKNNOptimizer, 40),
        (P3alphaOptimizer, 40),
        (RP3betaOptimizer, 40),
        (IALSOptimizer, 40),
        (DenseSLIMOptimizer, 20),
        # (SLIMOptimizer, 40), time consuming
        # (MultVAEOptimizer, 5),
    ]
    for optimizer_class, n_trials in test_configs:
        recommender_name = optimizer_class.recommender_class.__name__
        optimizer: BaseOptimizer = optimizer_class(data_train.X_all, valid_evaluator)
        (best_param, validation_result_df) = optimizer.optimize(
            timeout=14400, n_trials=n_trials
        )
        validation_result_df["recommender_name"] = recommender_name
        validation_results.append(validation_result_df)
        pd.concat(validation_results).to_csv(f"validation_scores.csv")
        test_recommender = optimizer.recommender_class(X_train_val_all, **best_param)
        test_recommender.learn()
        test_scores = test_evaluator.get_scores(test_recommender, [5, 10, 20])

        test_results.append(
            dict(name=recommender_name, best_param=best_param, **test_scores)
        )
        with open("test_results.json", "w") as ofs:
            json.dump(test_results, ofs, indent=2)
