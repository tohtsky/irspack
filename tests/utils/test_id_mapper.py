import uuid
from typing import Any, Dict, List, Tuple

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


@pytest.mark.parametrize("dtype", ["float32", "float64", "float16"])
def test_basic_usecase(dtype: str) -> None:
    RNS = np.random.RandomState(0)
    n_users = 31
    n_items = 42
    user_ids = [str(uuid.uuid4()) for _ in range(n_users)]
    item_ids = [str(uuid.uuid4()) for _ in range(n_items)]
    item_id_set = set(item_ids)
    score = RNS.randn(n_users, n_items).astype(dtype)
    X = sps.csr_matrix((score + RNS.randn(*score.shape)) > 0).astype(np.float64)
    rec = MockRecommender(X, score).learn()

    with pytest.raises(ValueError):
        mapped_rec = IDMappedRecommender(rec, user_ids, item_ids + [str(uuid.uuid4())])

    mapped_rec = IDMappedRecommender(rec, user_ids, item_ids)

    with pytest.raises(RuntimeError):
        mapped_rec.get_recommendation_for_known_user_id(str(uuid.uuid4()))

    known_user_results_individual = []
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
        recommendation_with_cutoff = mapped_rec.get_recommendation_for_known_user_id(
            uid, cutoff=cutoff
        )
        known_user_results_individual.append(recommendation_with_cutoff)
        check_descending(recommendation_with_cutoff)
        assert len(recommendation_with_cutoff) <= cutoff

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

    if dtype == "float16":
        with pytest.raises(ValueError):
            known_user_results_batch = (
                mapped_rec.get_recommendation_for_known_user_batch(
                    user_ids, cutoff=n_items
                )
            )
        return
    known_user_results_batch = mapped_rec.get_recommendation_for_known_user_batch(
        user_ids, cutoff=n_items
    )
    assert len(known_user_results_batch) == len(known_user_results_individual)
    for batch_res, indiv_res in zip(
        known_user_results_batch, known_user_results_individual
    ):
        for (id_batch, score_batch), (id_indiv, score_indiv) in zip(
            batch_res, indiv_res
        ):
            assert id_batch == id_indiv
            assert score_batch == pytest.approx(score_indiv)

    nonzero_batch: List[List[str]] = []
    forbidden_items_batch: List[List[str]] = []
    allowed_items_batch: List[List[str]] = []
    for i, _ in enumerate(user_ids):
        nonzero_items = [item_ids[j] for j in X[i].nonzero()[1]]
        nonzero_batch.append(nonzero_items)
        forbidden_items_batch.append(
            list(
                set(
                    RNS.choice(
                        list(item_id_set.difference(nonzero_items)),
                        replace=False,
                        size=(n_items - len(nonzero_items)) // 2,
                    )
                )
            )
        )
        allowed_items_batch.append(
            list(
                RNS.choice(
                    list(item_id_set.difference(nonzero_items)),
                    size=min(n_items - len(nonzero_items), n_items // 3),
                )
            )
        )
    batch_result_non_masked = mapped_rec.get_recommendation_for_new_user_batch(
        nonzero_batch, cutoff=n_items, n_threads=1
    )

    assert len(batch_result_non_masked) == n_users
    for recommendation_using_batch, uid, nonzero_items in zip(
        batch_result_non_masked, user_ids, nonzero_batch
    ):
        check_descending(recommendation_using_batch)
        recommended_ids = {rec[0] for rec in recommendation_using_batch}
        assert len(recommended_ids.intersection(nonzero_items)) == 0
        assert len(recommended_ids.union(nonzero_items)) == n_items

    batch_result_non_masked_cutoff_3 = mapped_rec.get_recommendation_for_new_user_batch(
        nonzero_batch, cutoff=3, n_threads=2
    )

    assert len(batch_result_non_masked_cutoff_3) == n_users
    for recommended_using_batch_cutoff_3, uid, nonzero_items in zip(
        batch_result_non_masked_cutoff_3, user_ids, nonzero_batch
    ):
        assert len(recommended_using_batch_cutoff_3) == 3
        check_descending(recommended_using_batch_cutoff_3)
        recommended_ids = {rec[0] for rec in recommended_using_batch_cutoff_3}
        for rid in recommended_ids:
            assert rid not in nonzero_items

    batch_result_masked = mapped_rec.get_recommendation_for_new_user_batch(
        nonzero_batch,
        cutoff=n_items,
        forbidden_item_ids=forbidden_items_batch,
        n_threads=1,
    )
    assert len(batch_result_masked) == n_users
    for recommended_using_batch_masked, uid, nonzero_items, forbidden_items in zip(
        batch_result_masked, user_ids, nonzero_batch, forbidden_items_batch
    ):
        check_descending(recommended_using_batch_masked)
        recommended_ids = {rec[0] for rec in recommended_using_batch_masked}
        assert len(recommended_ids.intersection(nonzero_items)) == 0
        assert (
            len(recommended_ids.union(nonzero_items).union(forbidden_items)) == n_items
        )

    batch_result_masked_restricted = mapped_rec.get_recommendation_for_new_user_batch(
        nonzero_batch,
        cutoff=n_items,
        forbidden_item_ids=forbidden_items_batch,
        per_user_allowed_item_ids=allowed_items_batch,
        n_threads=1,
    )
    assert len(batch_result_masked_restricted) == n_users
    for (
        recommended_using_batch_masked_restricted,
        uid,
        nonzero_items,
        forbidden_items,
        allowed_items,
    ) in zip(
        batch_result_masked_restricted,
        user_ids,
        nonzero_batch,
        forbidden_items_batch,
        allowed_items_batch,
    ):
        check_descending(recommended_using_batch_masked_restricted)
        recommended_ids = {rec[0] for rec in recommended_using_batch_masked_restricted}
        assert not recommended_ids.intersection(nonzero_items)
        for rid in recommended_ids:
            assert rid in allowed_items
            assert rid not in forbidden_items

    nonzero_batch_dict: List[Dict[str, float]] = []
    for i, _ in enumerate(user_ids):
        profile = {item_ids[j]: 2.0 for j in X[i].nonzero()[1]}
        nonzero_batch_dict.append(profile)
    batch_result_dict = mapped_rec.get_recommendation_for_new_user_batch(
        nonzero_batch_dict, cutoff=n_items, n_threads=1
    )
    for i, result in enumerate(batch_result_dict):
        nnz = len(X[i].nonzero()[1])
        softmax_denom = X.shape[1] - nnz + np.exp(2) * nnz
        for _, score in result:
            assert score == pytest.approx(1 / softmax_denom)

    allowed_items_uniform = [str(x) for x in RNS.choice(item_ids, size=2)]
    batch_result_masked_uniform_allowed_ids = (
        mapped_rec.get_recommendation_for_new_user_batch(
            nonzero_batch,
            cutoff=n_items,
            allowed_item_ids=allowed_items_uniform,
            n_threads=1,
        )
    )
    cnt = 0
    for x in batch_result_masked_uniform_allowed_ids:
        for rec_id, score in x:
            assert rec_id in allowed_items_uniform
            cnt += 1
    assert cnt > 0
