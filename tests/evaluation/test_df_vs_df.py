from uuid import uuid1

import pandas as pd

from irspack.evaluation.evaluate_df_to_df import evaluate_recommendation_df
from irspack.evaluation.evaluator import Evaluator
from irspack.recommenders.ials import IALSRecommender
from irspack.utils import IDMappedRecommender, rowwise_train_test_split
from irspack.utils.sample_data import mf_example_data


def test_matching() -> None:
    n_users = 20
    n_items = 15
    cutoff = 10
    user_column = "UserId"
    item_column = "ItemId"
    X = mf_example_data(n_users, n_items, random_state=0)
    user_ids = [str(uuid1()) for _ in range(n_users)]
    item_ids = [str(uuid1()) for _ in range(n_items)]
    X_train, X_test = rowwise_train_test_split(X, test_ratio=0.2, random_state=0)
    X_test_as_list = []
    for row, col in zip(*X_test.nonzero()):
        X_test_as_list.append({user_column: user_ids[row], item_column: item_ids[col]})
    X_test_as_df = pd.DataFrame(X_test_as_list)
    rec = IALSRecommender(X_train, n_components=2, max_epoch=3).learn()
    mrec = IDMappedRecommender(rec, user_ids, item_ids)

    rec_list = mrec.get_recommendation_for_known_user_batch(user_ids, cutoff=cutoff)
    recommemdation_list = []
    for uid, recommend_per_user in zip(user_ids, rec_list):
        for iid, _ in recommend_per_user:
            recommemdation_list.append({user_column: uid, item_column: iid})
    recommendation_df = pd.DataFrame(recommemdation_list)
    evaluator = Evaluator(X_test, cutoff=cutoff)
    score_using_evaluator = evaluator.get_score(rec)
    score_using_df_vs_df = evaluate_recommendation_df(
        recommendation_df, X_test_as_df, user_column, item_column
    )
    for key in score_using_df_vs_df:
        assert score_using_evaluator[key] == score_using_df_vs_df[key]
