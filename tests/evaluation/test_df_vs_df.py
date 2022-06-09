from uuid import uuid1

import numpy as np
import pandas as pd
from scipy import sparse as sps

from irspack.evaluation.evaluate_df_to_df import evaluate_recommendation_df
from irspack.evaluation.evaluator import Evaluator
from irspack.recommenders.ials import IALSRecommender
from irspack.utils import IDMappedRecommender, rowwise_train_test_split


def test_matching() -> None:
    n_users = 30
    n_items = 20
    cutoff = 3
    user_column = "UserId"
    item_column = "ItemId"
    _, X = rowwise_train_test_split(
        sps.csr_matrix(np.ones((n_users, n_items), dtype=np.float64)),
        n_test=n_users // 2,
        random_state=0,
    )
    assert np.all(X.data == 1)
    X_train, X_test = rowwise_train_test_split(X, random_state=0, n_test=3)
    assert X_train.multiply(X_test).getnnz() == 0
    del X
    user_ids = [str(uuid1()) for _ in range(n_users)]
    item_ids = [str(uuid1()) for _ in range(n_items)]

    def _sps_to_df(X: sps.csr_matrix) -> pd.DataFrame:
        X_as_list = []
        for row, col in zip(*X.nonzero()):
            X_as_list.append({user_column: user_ids[row], item_column: item_ids[col]})
        return pd.DataFrame(X_as_list)

    X_train_as_df = _sps_to_df(X_train)
    X_test_as_df = _sps_to_df(X_test)

    rec = IALSRecommender(X_train, n_components=1, max_epoch=0).learn()
    score = rec.get_score_remove_seen_block(0, n_users)
    mrec = IDMappedRecommender(rec, user_ids, item_ids)

    rec_list = mrec.get_recommendation_for_known_user_batch(user_ids, cutoff=cutoff)
    recommemdation_list = []
    for uid, recommend_per_user in zip(user_ids, rec_list):
        for iid, _ in recommend_per_user:
            recommemdation_list.append({user_column: uid, item_column: iid})
    recommendation_df = pd.DataFrame(recommemdation_list)
    assert recommendation_df.merge(X_train_as_df).shape[0] == 0
    evaluator = Evaluator(X_test, cutoff=cutoff, n_threads=1)
    score_using_evaluator = evaluator.get_score(rec)
    print("===== df to df =====")
    score_using_df_vs_df = evaluate_recommendation_df(
        recommendation_df, X_test_as_df, user_column, item_column, n_threads=1
    )
    assert score_using_evaluator["ndcg"] == score_using_df_vs_df["ndcg"]
    assert score_using_evaluator["map"] == score_using_df_vs_df["map"]
    assert score_using_evaluator["hit"] == score_using_df_vs_df["hit"]
    assert score_using_evaluator["recall"] == score_using_df_vs_df["recall"]
    assert score_using_evaluator["precision"] == score_using_df_vs_df["precision"]
