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
    DenseSLIMOptimizer,
    IALSOptimizer,
    MultVAEOptimizer,
    P3alphaOptimizer,
    RP3betaOptimizer,
    SLIMOptimizer,
    TopPopOptimizer,
)
from rs_evaluation.split import split

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["RS_THREAD_DEFAULT"] = "16"

if __name__ == "__main__":
    BASE_CUTOFF = 20

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler("search.log"))
    logger.addHandler(logging.StreamHandler())

    df_all = pd.read_csv("candidate_view_log.csv")
    cnt = df_all.job_id.value_counts()
    df_all = df_all[df_all.job_id.isin(cnt[cnt >= 10].index)]
    print(df_all.shape)
    data_all, _ = split(
        df_all, "candidate_id", "job_id", split_ratio_test=0.8, split_ratio_val=0.8
    )

    data_train = data_all["train"]
    data_val = data_all["val"]
    data_test = data_all["test"]

    X_train_all = sps.vstack([data_train.X_learn, data_val.X_learn, data_test.X_learn])
    X_train_val_all = sps.vstack([data_train.X_all, data_val.X_all, data_test.X_learn])

    valid_evaluator = Evaluator(
        ground_truth=data_val.X_predict, offset=data_train.n_users, cutoff=BASE_CUTOFF,
    )
    test_evaluator = Evaluator(
        ground_truth=data_test.X_predict,
        offset=(data_train.n_users + data_val.n_users),
        cutoff=BASE_CUTOFF,
    )
    test_results = []

    test_configs: List[Tuple[Type[BaseOptimizer], int]] = [
        (TopPopOptimizer, 1),
        (P3alphaOptimizer, 10),
        (RP3betaOptimizer, 40),
        (BPRFMOptimizer, 40),
        (DenseSLIMOptimizer, 10),
        (IALSOptimizer, 40),
        (MultVAEOptimizer, 5),
        (SLIMOptimizer, 40),
    ]
    for optimizer_class, n_trials in test_configs:
        name = optimizer_class.__name__
        optimizer = optimizer_class(
            X_train_all,
            valid_evaluator,
            metric="ndcg",
            n_trials=n_trials,
            logger=logger,
        )
        (best_param, validation_results) = optimizer.do_search(name, timeout=14400)
        validation_results.to_csv(f"{name}_validation_scores.csv")
        test_recommender = optimizer.recommender_class(X_train_val_all, **best_param)
        test_recommender.learn()
        test_scores = test_evaluator.get_scores(test_recommender, [5, 10, 20])

        test_results.append(
            dict(name=name, best_param=best_param, test_scores=test_scores)
        )
        with open("test_results.json", "w") as ofs:
            json.dump(test_results, ofs)
