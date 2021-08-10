import pickle
from collections import defaultdict
from io import BytesIO

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.evaluator import Evaluator, EvaluatorWithColdUser
from irspack.recommenders import P3alphaRecommender, TopPopRecommender
from irspack.recommenders.base import BaseRecommender
from irspack.split import rowwise_train_test_split


class MockRecommender(BaseRecommender, register_class=False):
    def __init__(self, X_all: sps.csr_matrix, scores: np.ndarray) -> None:
        super().__init__(X_all)
        self.scores = scores

    def get_score(self, user_indices: np.ndarray) -> np.ndarray:
        return self.scores[user_indices]

    def _learn(self) -> None:
        pass


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
