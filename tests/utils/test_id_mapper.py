import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.definitions import DenseScoreArray, InteractionMatrix
from irspack.recommenders import BaseRecommender
from irspack.utils.id_mapping import IDMapper, ItemIDMapper


class MockRecommender(BaseRecommender):
    def __init__(self, X_all: sps.csr_matrix, scores: np.ndarray) -> None:
        super().__init__(X_all)
        self.scores = scores

    def get_score(self, user_indices: np.ndarray) -> np.ndarray:
        score: np.ndarray = self.scores[user_indices]
        return score

    def _learn(self) -> None:
        pass

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        score: np.ndarray = np.exp(X.toarray())
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
        id_mapper_extra_item_ids = IDMapper(user_ids, item_ids + [str(uuid.uuid4())])
        id_mapper_extra_item_ids.recommend_for_known_user_id(rec, user_ids[0])

    with pytest.raises(ValueError):
        id_mapper_extra_user_ids = IDMapper(user_ids + [str(uuid.uuid4())], item_ids)
        id_mapper_extra_user_ids.recommend_for_known_user_id(rec, user_ids[0])

    id_mapper = IDMapper(user_ids, item_ids)

    with pytest.raises(RuntimeError):
        id_mapper.recommend_for_known_user_id(rec, str(uuid.uuid4()))

    known_user_results_individual = []
    for i, uid in enumerate(user_ids):
        nonzero_items = [item_ids[j] for j in X[i].nonzero()[1]]
        recommended = id_mapper.recommend_for_known_user_id(rec, uid, cutoff=n_items)
        check_descending(recommended)
        recommended_ids = {rec[0] for rec in recommended}
        assert len(recommended_ids.intersection(nonzero_items)) == 0
        assert len(recommended_ids.union(nonzero_items)) == n_items
        cutoff = n_items // 2
        recommendation_with_cutoff = id_mapper.recommend_for_known_user_id(
            rec, uid, cutoff=cutoff
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

        recommended_with_restriction = id_mapper.recommend_for_known_user_id(
            rec, uid, cutoff=n_items, forbidden_item_ids=forbbiden_item
        )
        check_descending(recommended_with_restriction)
        for iid, _ in recommended_with_restriction:
            assert iid not in forbbiden_item

        random_allowed_items = list(
            RNS.choice(
                list(item_id_set.difference(nonzero_items)),
                size=min(n_items - len(nonzero_items), n_items // 3),
            )
        )
        random_allowed_items += [str(uuid.uuid1())]
        with_allowed_item = id_mapper.recommend_for_known_user_id(
            rec,
            uid,
            cutoff=n_items,
            allowed_item_ids=random_allowed_items,
        )
        check_descending(with_allowed_item)
        for id_, _ in with_allowed_item:
            assert id_ in random_allowed_items

        allowed_only_forbidden = id_mapper.recommend_for_known_user_id(
            rec,
            uid,
            cutoff=n_items,
            forbidden_item_ids=forbbiden_item,
            allowed_item_ids=forbbiden_item,
        )
        assert not allowed_only_forbidden

        recommendations_for_coldstart = id_mapper.recommend_for_new_user(
            rec, nonzero_items, cutoff=n_items
        )
        coldstart_rec_results = {_[0] for _ in recommendations_for_coldstart}
        assert len(coldstart_rec_results.intersection(nonzero_items)) == 0
        assert len(coldstart_rec_results.union(nonzero_items)) == n_items

    if dtype == "float16":
        with pytest.raises(ValueError):
            known_user_results_batch = id_mapper.recommend_for_known_user_batch(
                rec, user_ids, cutoff=n_items
            )
        return
    known_user_results_batch = id_mapper.recommend_for_known_user_batch(
        rec, user_ids, cutoff=n_items
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
    for i, _ in enumerate(user_ids):
        nonzero_items = [item_ids[j] for j in X[i].nonzero()[1]]
        nonzero_batch.append(nonzero_items)

    batch_result_non_masked = id_mapper.recommend_for_new_user_batch(
        rec, nonzero_batch, cutoff=n_items, n_threads=1
    )

    assert len(batch_result_non_masked) == n_users
    for recommendation_using_batch, uid, nonzero_items in zip(
        batch_result_non_masked, user_ids, nonzero_batch
    ):
        check_descending(recommendation_using_batch)
        recommended_ids = {rec[0] for rec in recommendation_using_batch}
        assert len(recommended_ids.intersection(nonzero_items)) == 0
        assert len(recommended_ids.union(nonzero_items)) == n_items

    batch_result_non_masked_cutoff_3 = id_mapper.recommend_for_new_user_batch(
        rec, nonzero_batch, cutoff=3, n_threads=2
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

    batch_result_masked = id_mapper.recommend_for_new_user_batch(
        rec,
        nonzero_batch,
        cutoff=n_items,
        n_threads=1,
    )
    assert len(batch_result_masked) == n_users
    for recommended_using_batch_masked, uid, nonzero_items in zip(
        batch_result_masked, user_ids, nonzero_batch
    ):
        check_descending(recommended_using_batch_masked)
        recommended_ids = {rec[0] for rec in recommended_using_batch_masked}
        assert len(recommended_ids.intersection(nonzero_items)) == 0

    nonzero_batch_dict: List[Dict[str, float]] = []
    for i, _ in enumerate(user_ids):
        profile = {item_ids[j]: 2.0 for j in X[i].nonzero()[1]}
        nonzero_batch_dict.append(profile)
    batch_result_dict = id_mapper.recommend_for_new_user_batch(
        rec, nonzero_batch_dict, cutoff=n_items, n_threads=1
    )
    for i, result in enumerate(batch_result_dict):
        nnz = len(X[i].nonzero()[1])
        softmax_denom = X.shape[1] - nnz + np.exp(2) * nnz
        for _, val in result:
            assert val == pytest.approx(1 / softmax_denom)


def test_item_id_mapper_per_user_allowed_item() -> None:
    ids = [(i, f"{i}") for i in np.random.choice(10, size=10, replace=False) + 1]
    id_mapepr = ItemIDMapper(ids)
    N_users = 10
    score = np.random.random((N_users, len(ids)))
    per_user_allowed_item_ids = [
        list(set([random.choice(ids) for _ in range(random.randint(2, 5))]))
        for __ in range(N_users)
    ]
    recommendation = id_mapepr.score_to_recommended_items_batch(
        score, cutoff=5, per_user_allowed_item_ids=per_user_allowed_item_ids
    )
    assert len(recommendation) == score.shape[0]
    for i, (rec, allowed_item_ids_for_user) in enumerate(
        zip(recommendation, per_user_allowed_item_ids)
    ):
        previous_score: Optional[float] = None
        for ((id_int, id_str), score_value) in rec:
            assert (id_int, id_str) in allowed_item_ids_for_user
            item_index = ids.index((id_int, f"{id_int}"))
            assert score[i, item_index] == pytest.approx(score_value)
            if previous_score is not None:
                assert score_value <= previous_score
            previous_score = score_value


def test_item_id_mapper_uniform_allowed_item() -> None:
    ids = [(i, f"{i}") for i in np.random.choice(10, size=10, replace=False) + 1]
    id_mapepr = ItemIDMapper(ids)
    N_users = 10
    score = np.random.random((N_users, len(ids)))

    allowed_items_uniform = list(set([random.choice(ids) for _ in range(5)]))
    assert allowed_items_uniform
    recommendation = id_mapepr.score_to_recommended_items_batch(
        score, cutoff=5, allowed_item_ids=allowed_items_uniform
    )
    assert len(recommendation) == score.shape[0]
    for i, rec in enumerate(recommendation):
        previous_score: Optional[float] = None
        for ((id_int, id_str), score_value) in rec:
            assert (id_int, id_str) in allowed_items_uniform
            item_index = ids.index((id_int, f"{id_int}"))
            assert score[i, item_index] == pytest.approx(score_value)
            if previous_score is not None:
                assert score_value <= previous_score
            previous_score = score_value


def test_item_id_mapper_per_user_allowed_item_forbidden_items() -> None:
    ids: List[int] = list(range(10))
    per_user_allowed_item_ids = [
        [0, 1, 2, 1048576],
        [0, 1],
        [1, 2, -1],
    ]
    forbidden_items = [[0, 1, 2], [0], [1]]
    # so
    # Nothing can be recommended for user 0
    # 1 is the only choice for user 1
    # 2 is the only choice for user 2

    N_users = len(per_user_allowed_item_ids)
    assert len(forbidden_items) == N_users
    np.random.shuffle(ids)
    id_mapepr = ItemIDMapper(ids)
    score = np.random.random((len(forbidden_items), len(ids)))
    recommendation = id_mapepr.score_to_recommended_items_batch(
        score,
        cutoff=10,
        per_user_allowed_item_ids=per_user_allowed_item_ids,
        forbidden_item_ids=forbidden_items,
    )
    assert len(recommendation) == score.shape[0]

    user_0_recommendation = recommendation[0]
    assert not user_0_recommendation

    user_1_recommendation = recommendation[1]
    assert len(user_1_recommendation) == 1
    assert user_1_recommendation[0][0] == 1

    user_2_recommendation = recommendation[2]
    assert len(user_2_recommendation) == 1
    assert user_2_recommendation[0][0] == 2
