import numpy as np
import pytest
import scipy.sparse as sps

from irspack.dataset.movielens import MovieLens100KDataManager
from irspack.split import split_dataframe_partial_user_holdout
from irspack.split.random import UserTrainTestInteractionPair

RNS = np.random.RandomState(0)

df = MovieLens100KDataManager(force_download=True).read_interaction()


def test_user_level_split() -> None:
    dataset, mid_list = split_dataframe_partial_user_holdout(
        df,
        user_column="userId",
        item_column="movieId",
        n_test_user=30,
        n_val_user=30,
        heldout_ratio_val=0.3,
        heldout_ratio_test=0.5,
    )
    assert len(mid_list) == len(set(df.movieId))
    train = dataset["train"]
    train_invalid = UserTrainTestInteractionPair(
        train.user_ids, train.X_train[:, :-1], None
    )
    with pytest.raises(ValueError):
        train_invalid.concat(train)
    with pytest.raises(ValueError):
        _ = UserTrainTestInteractionPair(
            train.user_ids, train.X_train, train.X_train[1:]
        )

    val = dataset["val"]
    test = dataset["test"]
    assert train.X_test.count_nonzero() == 0
    train_val = train.concat(val)
    assert train_val.X_test[: train.n_users].count_nonzero() == 0
    assert (train_val.X_test[train.n_users :] - val.X_test).count_nonzero() == 0

    assert (
        train_val.X_train - sps.vstack([train.X_all, val.X_train])
    ).count_nonzero() == 0

    for user_data, ratio in [(val, 0.3), (test, 0.5)]:
        X_learn = user_data.X_train
        X_predict = user_data.X_test
        assert X_predict is not None
        intersect = X_learn.multiply(X_predict)
        assert intersect.count_nonzero() == 0
        index = RNS.choice(np.arange(user_data.n_users), size=10)
        for i in index:
            nnz_learn = X_learn[i].nonzero()[1].shape[0]
            nnz_predict = X_predict[i].nonzero()[1].shape[0]
            assert ratio >= (nnz_predict - 1) / (nnz_learn + nnz_predict)
            assert ratio <= (nnz_predict + 1) / (nnz_learn + nnz_predict)


def test_user_level_split_val_fixed_n() -> None:
    ratio = 0.3
    dataset, mid_list = split_dataframe_partial_user_holdout(
        df,
        user_column="userId",
        item_column="movieId",
        n_test_user=30,
        n_val_user=30,
        n_heldout_val=1,
        heldout_ratio_test=ratio,
    )
    assert len(mid_list) == len(set(df.movieId))

    train = dataset["train"]
    train_invalid = UserTrainTestInteractionPair(
        train.user_ids, train.X_train[:, :-1], None
    )
    with pytest.raises(ValueError):
        train_invalid.concat(train)
    with pytest.raises(ValueError):
        _ = UserTrainTestInteractionPair(
            train.user_ids, train.X_train, train.X_train[1:]
        )

    val = dataset["val"]
    test = dataset["test"]
    assert train.X_test.count_nonzero() == 0
    train_val = train.concat(val)
    assert train_val.X_test[: train.n_users].count_nonzero() == 0
    assert (train_val.X_test[train.n_users :] - val.X_test).count_nonzero() == 0

    assert (
        train_val.X_train - sps.vstack([train.X_all, val.X_train])
    ).count_nonzero() == 0

    val_X_test = val.X_test
    assert np.all(val_X_test.sum(axis=1).A1 <= 1)

    X_learn = test.X_train
    X_predict = test.X_test
    assert X_predict is not None
    intersect = X_learn.multiply(X_predict)
    assert intersect.count_nonzero() == 0
    index = RNS.choice(np.arange(test.n_users), size=10)
    for i in index:
        nnz_learn = X_learn[i].nonzero()[1].shape[0]
        nnz_predict = X_predict[i].nonzero()[1].shape[0]
        assert ratio >= (nnz_predict - 1) / (nnz_learn + nnz_predict)
        assert ratio <= (nnz_predict + 1) / (nnz_learn + nnz_predict)
