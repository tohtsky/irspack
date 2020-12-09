import json
import logging
import os
from typing import List, Tuple, Type

import pandas as pd
from scipy import sparse as sps

from irspack.evaluator import EvaluatorWithColdUser
from irspack.dataset.movielens import MovieLens100KDataManager
from irspack.optimizers import (
    BaseOptimizer,
    DenseSLIMOptimizer,
    IALSOptimizer,
    MultVAEOptimizer,
    P3alphaOptimizer,
    RP3betaOptimizer,
    SLIMOptimizer,
    TopPopOptimizer,
)
from irspack.split import split

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["RS_THREAD_DEFAULT"] = "4"

if __name__ == "__main__":

    BASE_CUTOFF = 20

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(f"search.log"))
    logger.addHandler(logging.StreamHandler())

    data_manager = MovieLens100KDataManager()
    df_all = data_manager.load_rating()

    data_all, _ = split(
        df_all, "movieId", "userId", split_ratio_test=0.5, split_ratio_val=0.5,
    )

    data_train = data_all["train"]
    data_val = data_all["val"]
    data_test = data_all["test"]

    X_train_val_all: sps.csr_matrix = sps.vstack(
        [data_train.X_all, data_val.X_all], format="csr"
    )
    valid_evaluator = EvaluatorWithColdUser(
        input_interaction=data_val.X_learn,
        ground_truth=data_val.X_predict,
        cutoff=BASE_CUTOFF,
    )
    test_evaluator = EvaluatorWithColdUser(
        input_interaction=data_test.X_learn,
        ground_truth=data_test.X_predict,
        cutoff=BASE_CUTOFF,
    )

    test_results = []
    validation_results = []

    test_configs: List[Tuple[Type[BaseOptimizer], int]] = [
        (TopPopOptimizer, 1),
        (IALSOptimizer, 40),
        (DenseSLIMOptimizer, 10),
        (P3alphaOptimizer, 10),
        (RP3betaOptimizer, 40),
        (MultVAEOptimizer, 5),
        (SLIMOptimizer, 40),
    ]
    for optimizer_class, n_trial in test_configs:
        recommender_name = optimizer_class.recommender_class.__name__
        optimizer: BaseOptimizer = optimizer_class(
            data_train.X_all,
            valid_evaluator,
            metric="ndcg",
            n_trials=n_trial,
            logger=logger,
        )
        (best_param, validation_result_df) = optimizer.do_search(timeout=14400)
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
