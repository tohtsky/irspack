import json
import logging
import os
from typing import List, Tuple, Type

import pandas as pd
from scipy import sparse as sps

from rs_evaluation.evaluator import Evaluator
from rs_evaluation.optimizers import (
    BaseOptimizer,
    BPRFMOptimizer,
    IALSOptimizer,
    MultVAEOptimizer,
    P3alphaOptimizer,
    RP3betaOptimizer,
    SLIMOptimizer,
    TopPopOptimizer,
)
from rs_evaluation.split import split

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["RS_THREAD_DEFAULT"] = "4"

if __name__ == "__main__":
    BASE_CUTOFF = 20

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(f"search.log"))
    logger.addHandler(logging.StreamHandler())

    df_all = pd.read_pickle("sessions_from_20180831.pkl")
    cnt = df_all.fullpath.value_counts()
    df_all = df_all[df_all.fullpath.isin(cnt[cnt >= 5].index)]
    data_all, _ = split(
        df_all, "user_id", "fullpath", split_ratio_test=0.8, split_ratio_val=0.8
    )
    train_data = data_all["train"]
    val_data = data_all["val"]
    test_data = data_all["test"]

    X_train_all = sps.vstack([train_data.X_learn, val_data.X_learn, test_data.X_learn])
    X_train_val_all = sps.vstack([train_data.X_all, val_data.X_all, test_data.X_learn])

    valid_evaluator = Evaluator(
        ground_truth=val_data.X_predict, offset=train_data.n_users, cutoff=BASE_CUTOFF,
    )
    test_evaluator = Evaluator(
        ground_truth=test_data.X_predict,
        offset=train_data.n_users + val_data.n_users,
        cutoff=BASE_CUTOFF,
    )
    test_configs: List[Tuple[Type[BaseOptimizer], int]] = [
        (TopPopOptimizer, 1),
        (P3alphaOptimizer, 10),
        (RP3betaOptimizer, 40),
        (IALSOptimizer, 40),
        (BPRFMOptimizer, 40),
        (MultVAEOptimizer, 5),
        (SLIMOptimizer, 40),
    ]
    test_results = []
    for optimizer_class, n_trial in test_configs:
        name = optimizer_class.__name__
        optimizer = optimizer_class(
            X_train_all, valid_evaluator, metric="ndcg", n_trials=n_trial, logger=logger
        )
        best_param, validation_results = optimizer.do_search(name, timeout=14400)
        validation_results.to_csv(f"{name}_validation_scores.csv")
        test_recommender = optimizer.recommender_class(X_train_val_all, **best_param)
        test_recommender.learn()
        test_scores = test_evaluator.get_scores(test_recommender, [5, 10, 20])
        test_results.append(
            dict(name=name, best_param=best_param, test_scores=test_scores)
        )
        with open("test_results.json", "w") as ofs:
            json.dump(test_results, ofs)
