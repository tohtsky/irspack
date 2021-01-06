import uuid
from typing import Any, List, Tuple

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.definitions import DenseScoreArray, InteractionMatrix
from irspack.recommenders import BaseRecommender
from irspack.utils.id_mapping import IDMappedRecommender


class MockRecommender(BaseRecommender):
    def __init__(self, X_all: sps.csr_matrix, scores: np.ndarray) -> None:
        super().__init__(X_all)
        self.scores = scores

    def get_score(self, user_indices: np.ndarray) -> np.ndarray:
        return self.scores[user_indices]

    def _learn(self) -> None:
        pass

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        score = np.exp(X.toarray())
        score /= score.sum(axis=1)[:, None]
        return score


def check_descending(id_score: List[Tuple[Any, float]]) -> None:
    for i, (_, score) in enumerate(id_score):
        if i > 0:
            if score > id_score[i - 1][1]:
                raise RuntimeError("not descending!")


def test_check_descending_1() -> None:
    check_descending([("1", 100), ("2", 99), ("100", 99), ("3", -1000)])
    with pytest.raises(RuntimeError):
        check_descending([("1", 100), ("2", 99), ("3", -1000), ("", -999)])


def test_basic_usecase() -> None:
    RNS = np.random.RandomState(0)
    n_users = 31
    n_items = 42
    user_ids = [str(uuid.uuid4()) for _ in range(n_users)]
    item_ids = [str(uuid.uuid4()) for _ in range(n_items)]
    item_id_set = set(item_ids)
    score = RNS.randn(n_users, n_items)
    X = sps.csr_matrix((score + RNS.randn(*score.shape)) > 0).astype(np.float64)
    rec = MockRecommender(X, score).learn()

    with pytest.raises(ValueError):
        mapped_rec = IDMappedRecommender(rec, user_ids, item_ids + [str(uuid.uuid4())])

    mapped_rec = IDMappedRecommender(rec, user_ids, item_ids)

    with pytest.raises(RuntimeError):
        mapped_rec.get_recommendation_for_known_user_id(str(uuid.uuid4()))

    for i, uid in enumerate(user_ids):
        nonzero_items = [item_ids[j] for j in X[i].nonzero()[1]]
        recommended = mapped_rec.get_recommendation_for_known_user_id(
            uid, cutoff=n_items
        )
        check_descending(recommended)
        recommended_ids = {rec[0] for rec in recommended}
        assert len(recommended_ids.intersection(nonzero_items)) == 0
        assert len(recommended_ids.union(nonzero_items)) == n_items
        cutoff = n_items // 2
        recommended_with_cutoff = mapped_rec.get_recommendation_for_known_user_id(
            uid, cutoff=cutoff
        )
        check_descending(recommended_with_cutoff)
        assert len(recommended_with_cutoff) <= cutoff

        forbbiden_item = list(
            set(
                RNS.choice(
                    list(item_id_set.difference(nonzero_items)),
                    replace=False,
                    size=(n_items - len(nonzero_items)) // 2,
                )
            )
        )

        recommended_with_restriction = mapped_rec.get_recommendation_for_known_user_id(
            uid, cutoff=n_items, forbidden_item_ids=forbbiden_item
        )
        check_descending(recommended_with_restriction)
        for iid, score in recommended_with_restriction:
            assert iid not in forbbiden_item

        random_allowed_items = list(
            RNS.choice(
                list(item_id_set.difference(nonzero_items)),
                size=min(n_items - len(nonzero_items), n_items // 3),
            )
        )
        random_allowed_items += [str(uuid.uuid1())]
        with_allowed_item = mapped_rec.get_recommendation_for_known_user_id(
            uid,
            cutoff=n_items,
            allowed_item_ids=random_allowed_items,
        )
        check_descending(with_allowed_item)
        for id, _ in with_allowed_item:
            assert id in random_allowed_items

        allowed_only_forbidden = mapped_rec.get_recommendation_for_known_user_id(
            uid,
            cutoff=n_items,
            forbidden_item_ids=forbbiden_item,
            allowed_item_ids=forbbiden_item,
        )
        assert not allowed_only_forbidden

        recommendations_for_coldstart = mapped_rec.get_recommendation_for_new_user(
            nonzero_items, cutoff=n_items
        )
        coldstart_rec_results = {_[0] for _ in recommendations_for_coldstart}
        assert len(coldstart_rec_results.intersection(nonzero_items)) == 0
        assert len(coldstart_rec_results.union(nonzero_items)) == n_items
