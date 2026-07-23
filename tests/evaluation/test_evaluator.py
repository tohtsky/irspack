import pickle
from collections import defaultdict
from io import BytesIO

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.evaluation import Evaluator, EvaluatorWithColdUser
from irspack.recommenders import P3alphaRecommender, TopPopRecommender
from irspack.recommenders.base import BaseRecommender
from irspack.split import rowwise_train_test_split

from ..mock_recommender import MockRecommender


@pytest.mark.parametrize(
    "U, I, dtype", [(10, 5, "float32"), (10, 30, "float64"), (3000, 5, "float32")]
)
def test_metrics(U: int, I: int, dtype: str) -> None:
    try:
        from sklearn.metrics import average_precision_score, ndcg_score
    except:
        pytest.skip()

    rns = np.random.RandomState(42)
    scores = rns.randn(U, I).astype(dtype)
    X_gt = (rns.rand(U, I) >= 0.7).astype(np.float64)
    eval = Evaluator(sps.csr_matrix(X_gt), offset=0, cutoff=I, n_threads=4)
    mock_rec = MockRecommender(sps.csr_matrix(X_gt.shape), scores)

    # float 16 not supported
    mock_rec_invalid = MockRecommender(
        sps.csr_matrix(X_gt.shape), scores.astype(np.float16)
    )
    with pytest.raises(ValueError):
        eval.get_score(mock_rec_invalid)
    my_score = eval.get_score(mock_rec)
    sklearn_metrics = defaultdict(list)
    for i in range(scores.shape[0]):
        if X_gt[i].sum() == 0:
            continue
        sklearn_metrics["map"].append(average_precision_score(X_gt[i], scores[i]))
        sklearn_metrics["ndcg"].append(ndcg_score(X_gt[i][None, :], scores[i][None, :]))

    for key in ["map", "ndcg"]:
        assert my_score[key] == pytest.approx(np.mean(sklearn_metrics[key]), abs=1e-8)

    with pytest.raises(ValueError):
        eval_emptymask = Evaluator(
            sps.csr_matrix(X_gt),
            offset=0,
            cutoff=I,
            n_threads=None,
            masked_interactions=sps.csr_matrix(
                (X_gt.shape[0] + 1, X_gt.shape[1])
            ),  # empty
        )

    X_gt = X_gt[(X_gt.sum(axis=1) > 0) & ((X_gt > 0).sum(axis=1) < I)]

    eval_emptymask = Evaluator(
        sps.csr_matrix(X_gt),
        offset=0,
        cutoff=1,
        n_threads=None,
        mb_size=3,
        masked_interactions=sps.csr_matrix(X_gt.shape),  # empty
        recall_with_cutoff=True,
    )
    mock_rec = MockRecommender(sps.csr_matrix(X_gt), X_gt)
    perfect_score = eval_emptymask.get_score(mock_rec)
    assert perfect_score["recall"] == pytest.approx(1.0)
    eval_vicious = Evaluator(
        sps.csr_matrix(X_gt),
        offset=0,
        cutoff=1,
        n_threads=1,
        masked_interactions=X_gt,
    )
    vicious_score = eval_vicious.get_score(mock_rec)
    assert vicious_score["recall"] == 0.0


@pytest.mark.parametrize("U, I, C", [(10, 5, 5), (10, 30, 29)])
def test_metrics_with_cutoff(U: int, I: int, C: int) -> None:
    try:
        from sklearn.metrics import ndcg_score
    except:
        pytest.skip()

    rns = np.random.RandomState(42)
    scores = rns.randn(U, I)
    X_gt = (rns.rand(U, I) >= 0.3).astype(np.float64)
    eval = Evaluator(sps.csr_matrix(X_gt), offset=0, cutoff=C, n_threads=2)
    eval_finer_chunk = Evaluator(
        sps.csr_matrix(X_gt), offset=0, cutoff=C, n_threads=2, mb_size=1
    )
    # empty mask
    mock_rec = MockRecommender(sps.csr_matrix(X_gt.shape), scores)
    my_score = eval.get_score(mock_rec)
    my_score_finer = eval_finer_chunk.get_score(mock_rec)
    for key in my_score:
        assert my_score_finer[key] == pytest.approx(my_score[key])

    ndcg = 0.0
    valid_users = 0
    map = 0.0
    precision = 0.0
    recall = 0.0
    item_appearance_count = np.zeros((I,), dtype=np.float64)
    for i in range(U):
        nzs = set(X_gt[i].nonzero()[0])
        if len(nzs) == 0:
            continue
        valid_users += 1
        ndcg += ndcg_score(X_gt[[i]], scores[[i]], k=C)
        recommended = scores[i].argsort()[::-1][:C]
        recall_denom = min(C, len(nzs))
        ap = 0.0
        current_hit = 0
        for i, rec in enumerate(recommended):
            item_appearance_count[rec] += 1.0
            if rec in nzs:
                current_hit += 1
                ap += current_hit / float(i + 1)
        ap /= recall_denom
        map += ap
        recall += current_hit / recall_denom
        precision += current_hit / C
    entropy = (lambda p: -p.dot(np.log(p)))(
        item_appearance_count / item_appearance_count.sum()
    )
    item_appearance_sorted_normalized = (
        np.sort(item_appearance_count) / item_appearance_count.sum()
    )
    lorentz_curve = np.cumsum(item_appearance_sorted_normalized)

    gini_index = 0
    delta = 1 / I
    for i in range(I):
        f = 2 * (((i + 1) / I) - lorentz_curve[i])
        gini_index += delta * f

    assert my_score["ndcg"] == pytest.approx(ndcg / valid_users)
    assert my_score["map"] == pytest.approx(map / valid_users, abs=1e-8)
    assert my_score["precision"] == pytest.approx(precision / valid_users, abs=1e-8)
    assert my_score["recall"] == pytest.approx(recall / valid_users, abs=1e-8)
    assert my_score["entropy"] == pytest.approx(entropy)
    assert my_score["gini_index"] == pytest.approx(gini_index)


@pytest.mark.parametrize("U, I, U_test", [(10, 5, 3), (10, 30, 8)])
def test_metrics_colduser_mask(U: int, I: int, U_test: int) -> None:
    rns = np.random.RandomState(42)
    X_gt = (rns.rand(U, I) >= 0.5).astype(np.float64)
    X_gt = X_gt[(X_gt.sum(axis=1) > 0)]
    X_gt = sps.csr_matrix(X_gt)
    rec = TopPopRecommender(X_gt).learn()
    vicious_eval = EvaluatorWithColdUser(X_gt, X_gt, cutoff=1)
    vicious_metric = vicious_eval.get_score(rec)
    assert vicious_metric["hit"] == 0.0

    popularity = rec.get_score_cold_user(X_gt[:1, :]).ravel()
    most_pop_indices = np.where(popularity.max() == popularity)[0]

    X_gt_pop = np.zeros(X_gt.shape)
    X_gt_pop[:, most_pop_indices] = 1
    X_gt_pop = sps.csr_matrix(X_gt_pop)
    generous_eval = EvaluatorWithColdUser(
        X_gt_pop,
        X_gt_pop,
        cutoff=1,
        masked_interactions=sps.csr_matrix(X_gt.shape),
        recall_with_cutoff=True,
    )
    generous_metric = generous_eval.get_score(rec)
    assert generous_metric["recall"] == 1.0

    pickle_content = BytesIO()
    pickle.dump(generous_eval, pickle_content)

    pickle_content.seek(0)

    generous_eval_pickled = pickle.load(pickle_content)

    assert generous_eval_pickled.get_score(rec)["recall"] == 1.0


@pytest.mark.parametrize("U, I, U_test", [(10, 5, 3), (10, 30, 8)])
def test_metrics_ColdUser(U: int, I: int, U_test: int) -> None:
    rns = np.random.RandomState(42)
    uvec = rns.randn(U + U_test, 3)
    ivec = rns.randn(I, 3)
    true_score = uvec.dot(ivec.T)  # + rns.randn(U, I)
    X = sps.csr_matrix((true_score > 0).astype(np.float64))
    X_train = X[:U]
    X_val = X[U:]
    X_val_learn, X_val_target = rowwise_train_test_split(X_val, random_state=0)
    X_train_all = sps.vstack([X_train, X_val_learn])
    hot_evaluator = Evaluator(
        sps.csr_matrix(X_val_target), offset=U, cutoff=I // 2, n_threads=2
    )

    rec = P3alphaRecommender(X_train_all)
    rec.learn()
    hot_score = hot_evaluator.get_score(rec)
    with pytest.warns(UserWarning):
        cold_evaluator = EvaluatorWithColdUser(
            X_val_learn.tocsc(), X_val_target, cutoff=I // 2, mb_size=5
        )  # csc matrix input should raise warning about
        # memory ordering, as csc-csc matrix product will be csc,
        # hence col-major matrix when made dense.
        cold_score = cold_evaluator.get_score(rec)

    shuffle_index = np.arange(X_val_learn.shape[0])
    rns.shuffle(shuffle_index)
    cold_evaluator_shuffled = EvaluatorWithColdUser(
        X_val_learn[shuffle_index], X_val_target[shuffle_index], cutoff=I // 2
    )
    cold_score_shuffled = cold_evaluator_shuffled.get_score(rec)
    for key in cold_score:
        assert cold_score_shuffled[key] == pytest.approx(cold_score[key])

    for key in hot_score:
        assert hot_score[key] == pytest.approx(cold_score[key], abs=1e-8)


@pytest.mark.parametrize("U, I, C", [(10, 5, 5), (10, 30, 29)])
def test_recommender_check(U: int, I: int, C: int) -> None:
    rns = np.random.RandomState(42)
    scores = rns.randn(U, I)
    X_gt = (rns.rand(U, I) >= 0.3).astype(np.float64)
    eval = Evaluator(sps.csr_matrix(X_gt), offset=0, cutoff=C, n_threads=2)
    mock_rec_too_few_users = MockRecommender(sps.csr_matrix((U - 1, I)), scores[1:])
    with pytest.raises(ValueError):
        eval.get_score(mock_rec_too_few_users)
    mock_rec_too_few_items = MockRecommender(sps.csr_matrix((U, I - 1)), scores[:, 1:])
    with pytest.raises(ValueError):
        eval.get_score(mock_rec_too_few_items)
    moc_rec_valid = MockRecommender(sps.csr_matrix((U, I)), scores)
    eval.get_score(moc_rec_valid)


def test_score_from_score_matrix() -> None:
    scores = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    original_scores = scores.copy()
    ground_truth = sps.csr_matrix([[0, 1], [1, 0]])
    mask = sps.csr_matrix([[1, 0], [0, 0]])
    evaluator = Evaluator(ground_truth, cutoff=1, masked_interactions=mask, mb_size=1)

    assert evaluator.get_score_from_score_matrix(scores)["recall"] == 1.0
    assert evaluator.get_scores_from_score_matrix(scores, [1])["recall@1"] == 1.0
    np.testing.assert_array_equal(scores, original_scores)

    with pytest.raises(ValueError, match="shape"):
        evaluator.get_score_from_score_matrix(scores[:, :1])
    with pytest.raises(ValueError, match="dtype"):
        evaluator.get_score_from_score_matrix(scores.astype(np.float16))


def test_score_from_score_matrix_cold_user_masks_input() -> None:
    input_interaction = sps.csr_matrix([[1, 0]])
    ground_truth = sps.csr_matrix([[0, 1]])
    evaluator = EvaluatorWithColdUser(input_interaction, ground_truth, cutoff=1)

    # The seen item has the highest raw score and must be excluded.
    assert (
        evaluator.get_score_from_score_matrix(np.array([[1.0, 0.0]], dtype=np.float64))[
            "recall"
        ]
        == 1.0
    )


def test_cold_user_evaluator_with_cold_item_features() -> None:
    class FeatureItemMockRecommender(BaseRecommender, register_class=False):
        def __init__(self) -> None:
            super().__init__(sps.csr_matrix((2, 2)))
            self.prepare_count = 0

        def _learn(self) -> None:
            pass

        def get_score(self, user_indices: np.ndarray) -> np.ndarray:
            return np.repeat(
                np.array([[0.2, 0.1]], dtype=np.float32),
                len(user_indices),
                axis=0,
            )

        def get_score_cold_user(self, X: sps.csr_matrix) -> np.ndarray:
            return np.repeat(
                np.array([[0.2, 0.1]], dtype=np.float32),
                X.shape[0],
                axis=0,
            )

        def _create_cold_user_with_item_features_scorer(self, item_features):
            self.prepare_count += 1
            np.testing.assert_array_equal(
                item_features, np.array([[1.0], [2.0]], dtype=np.float32)
            )

            def scorer(X):
                return np.repeat(
                    np.array([[0.2, 0.1, 0.9, 0.8]], dtype=np.float32),
                    X.shape[0],
                    axis=0,
                )

            return scorer

    input_interaction = sps.csr_matrix([[1, 0], [0, 1]], dtype=np.float32)
    ground_truth = sps.csr_matrix([[0, 0, 1, 0], [0, 0, 1, 0]])
    cold_item_features = np.array([[1.0], [2.0]], dtype=np.float32)
    evaluator = EvaluatorWithColdUser(
        input_interaction,
        ground_truth,
        cold_item_features=cold_item_features,
        cutoff=1,
        mb_size=1,
        n_threads=1,
    )
    rec = FeatureItemMockRecommender()

    score = evaluator.get_score(rec)
    assert score["recall"] == 1.0
    assert score["catalog_coverage"] == 0.25
    assert rec.prepare_count == 1
    qualified_score = evaluator.get_scores(rec, [1])
    assert qualified_score["catalog_coverage@1"] == 0.25

    fallback = TopPopRecommender(input_interaction).learn()
    fallback_score = evaluator.get_score(fallback)
    assert fallback_score["recall"] == 0.0
    # Negative-infinity cold-item scores are not counted as recommendations.
    assert fallback_score["appeared_item"] == 2.0


def test_cold_user_evaluator_cold_item_shape_validation() -> None:
    input_interaction = sps.csr_matrix((2, 3))
    ground_truth = sps.csr_matrix((2, 4))
    cold_item_features = np.ones((2, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="ground_truth"):
        EvaluatorWithColdUser(
            input_interaction,
            ground_truth,
            cold_item_features=cold_item_features,
        )


def test_negative_infinity_scores_are_not_recommendations() -> None:
    ground_truth = sps.csr_matrix([[1, 0, 0]])
    evaluator = Evaluator(ground_truth, cutoff=3)
    score = evaluator.get_score_from_score_matrix(
        np.array([[1.0, -np.inf, -np.inf]], dtype=np.float64)
    )

    assert score["recall"] == 1.0
    assert score["precision"] == 1.0
    assert score["appeared_item"] == 1.0
    assert score["catalog_coverage"] == pytest.approx(1 / 3)


def test_score_from_score_chunks_matches_matrix() -> None:
    rns = np.random.RandomState(0)
    U, I = 11, 7
    scores = rns.randn(U, I).astype(np.float64)
    original_scores = scores.copy()
    X_gt = sps.csr_matrix((rns.rand(U, I) >= 0.5).astype(np.float64))
    mask = sps.csr_matrix((rns.rand(U, I) >= 0.5).astype(np.float64))
    evaluator = Evaluator(X_gt, cutoff=3, masked_interactions=mask, mb_size=2)

    expected = evaluator.get_scores_from_score_matrix(scores, [1, 3])

    # Several non-uniform chunk sizes; total still U.
    split_points = [0, 1, 1, 3, 3, 6, 10, 11]
    chunks = [
        scores[split_points[i] : split_points[i + 1]]
        for i in range(len(split_points) - 1)
        if split_points[i] != split_points[i + 1]
    ]
    got = evaluator.get_scores_from_score_chunks(iter(chunks), [1, 3])
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, abs=1e-12), key
    np.testing.assert_array_equal(scores, original_scores)

    # chunked input must not mutate caller arrays either
    seen = [c.copy() for c in chunks]
    evaluator.get_scores_from_score_chunks(iter(chunks), [3])
    for original, mutated in zip(seen, chunks):
        np.testing.assert_array_equal(original, mutated)


def test_score_from_score_chunks_cold_user_masks_input() -> None:
    input_interaction = sps.csr_matrix([[1, 0]])
    ground_truth = sps.csr_matrix([[0, 1]])
    evaluator = EvaluatorWithColdUser(input_interaction, ground_truth, cutoff=1)

    assert (
        evaluator.get_score_from_score_chunks(
            iter([np.array([[1.0, 0.0]], dtype=np.float64)])
        )["recall"]
        == 1.0
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_score_from_score_chunks_errors(dtype: type) -> None:
    U, I = 3, 4
    X_gt = sps.csr_matrix(np.eye(U, I, dtype=np.float64))
    evaluator = Evaluator(X_gt, cutoff=2)

    # wrong number of columns
    with pytest.raises(ValueError, match="n_items"):
        evaluator.get_score_from_score_chunks(iter([np.zeros((U, I - 1), dtype=dtype)]))
    # wrong dtype
    with pytest.raises(ValueError, match="dtype"):
        evaluator.get_score_from_score_chunks(
            iter([np.zeros((U, I), dtype=np.float16)])
        )
    # too few rows
    with pytest.raises(ValueError, match="did not cover"):
        evaluator.get_score_from_score_chunks(iter([np.zeros((U - 1, I), dtype=dtype)]))
    # too many rows
    with pytest.raises(ValueError, match="more rows"):
        evaluator.get_score_from_score_chunks(iter([np.zeros((U + 1, I), dtype=dtype)]))
    # not a 2-D array (1-D ndarray slipped into the iterator)
    with pytest.raises(ValueError, match="2-D ndarray"):
        evaluator.get_score_from_score_chunks(iter([np.zeros(I, dtype=dtype)]))
