import os
import pickle
from typing import List, Tuple, Type
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import sparse as sps
import logging

from rs_evaluation.user_cold_start.cb2cf import (
    CB2BPRFMOptimizer,
    CB2TruncatedSVDOptimizer,
    CB2IALSOptimizer,
)
from rs_evaluation.user_cold_start.base import UserColdStartRecommenderBase
from rs_evaluation.user_cold_start.linear import LinearRecommender
from rs_evaluation.user_cold_start.popular import TopPopularRecommender
from rs_evaluation.user_cold_start.evaluator import UserColdStartEvaluator


os.environ["OMP_NUM_THREADS"] = "16"
os.environ["RS_THREAD_DEFAULT"] = "16"

if __name__ == "__main__":
    CUTOFF = 20
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(f"search.log"))
    logger.addHandler(logging.StreamHandler())

    # configuration for Target List.
    USER_COLNAME = "job_id"
    ITEM_COLNAME = "candidate_id"

    with open("coldstart_dataset_op.pkl", "rb") as ifs:
        d = pickle.load(ifs)
    X_profile = d["user_profile"]
    df = d["interaction_df"]
    user_ids = d["user_ids"]

    item_cnt = df[ITEM_COLNAME].value_counts()
    item_cnt = item_cnt[item_cnt >= 10]
    df = df[df[ITEM_COLNAME].isin(item_cnt.index)]

    itemid_all = np.unique(df[ITEM_COLNAME])
    item_id_to_index = {iid: i for i, iid in enumerate(itemid_all)}

    user_id_to_profile_index = {uid: i for i, uid in enumerate(user_ids)}
    train_users, test_users = train_test_split(user_ids, test_size=0.2, random_state=42)

    df_train = df[(df[USER_COLNAME].isin(train_users))]
    df_test = df[(df[USER_COLNAME].isin(test_users))]

    def to_matrix(df):
        user_ids, user_index = np.unique(df[USER_COLNAME], return_inverse=True)
        item_index = df[ITEM_COLNAME].map(item_id_to_index)
        return (
            user_ids,
            sps.csr_matrix(
                ((np.ones(df.shape[0], dtype=np.int32), (user_index, item_index))),
                shape=(user_ids.shape[0], len(item_id_to_index)),
            ),
        )

    user_ids_train, X_train = to_matrix(df_train)
    user_ids_test, X_test = to_matrix(df_test)

    profile_train = X_profile[
        [user_id_to_profile_index[uid] for uid in user_ids_train]
    ].copy()
    profile_test = X_profile[
        [user_id_to_profile_index[uid] for uid in user_ids_test]
    ].copy()

    test_evaluator = UserColdStartEvaluator(X_test, profile_test)
    test_results = []
    recommender_and_trials: List[Tuple[Type[UserColdStartRecommenderBase], int]] = [
        (TopPopularRecommender, 1),
        (LinearRecommender, 10),
    ]
    for recommender_class, n_trials in recommender_and_trials:
        best_config = recommender_class.optimize(
            X_train, profile_train, n_trials, target_metric="ndcg", timeout=14400
        )
        rec = recommender_class(X_train, profile_train, **best_config)
        rec.learn()
        metric = test_evaluator.get_score(rec, CUTOFF)
        logger.info(f"{recommender_class.__name__}: {metric}")
        metric["algorithm"] = recommender_class.__name__
        test_results.append(metric)
        pd.DataFrame(test_results).to_csv("result.csv")

    # CB2CF types
    for optimizer_class, n_trials in [
        (CB2TruncatedSVDOptimizer, 10),
        (CB2BPRFMOptimizer, 40),
        (CB2IALSOptimizer, 40),
    ]:

        optimizer = optimizer_class(X_train, profile_train)
        cold_recommender = optimizer.search_all(
            n_trials=n_trials, logger=logger, timeout=14400
        )
        metric = test_evaluator.get_score(cold_recommender, CUTOFF)
        test_results.append(metric)
        logger.info(f"{optimizer_class.__name__}: {metric}")
        metric["algorithm"] = optimizer_class.__name__
        pd.DataFrame(test_results).to_csv("result.csv")

