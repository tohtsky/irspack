from typing import Optional

import numpy as np
import pytest
import scipy.sparse as sps

from irspack.dataset.movielens import MovieLens100KDataManager
from irspack.split import (
    UserTrainTestInteractionPair,
    split_dataframe_partial_user_holdout,
)

RNS = np.random.RandomState(0)

df = MovieLens100KDataManager(force_download=True).read_interaction()

test_configs = [
    (None, None, 0.1, 0.15),
    (None, 3, 0.2, 0.25),
    (4, None, 0.3, 0.35),
    (5, 6, 0.0, 1.0),
]
test_configs_with_time = [
    config + (tsc,) for tsc in ["timestamp", None] for config in test_configs
]

test_configs_with_time_and_ceil_flg = []

for ceil_ in [True, False]:
    test_configs_with_time_and_ceil_flg.extend(
        [x + (ceil_,) for x in test_configs_with_time]
    )


@pytest.mark.parametrize(
    "n_val_user, n_test_user, val_user_ratio, test_user_ratio, time_colname, ceil",
    test_configs_with_time_and_ceil_flg,
)
def test_user_level_split(
    n_val_user: Optional[int],
    n_test_user: Optional[int],
    val_user_ratio: float,
    test_user_ratio: float,
    time_colname: Optional[str],
    ceil: bool,
) -> None:
    n_users_all = len(set(df.userId))
    dataset, mid_list = split_dataframe_partial_user_holdout(
        df,
        user_column="userId",
        item_column="movieId",
        time_column=time_colname,
        val_user_ratio=val_user_ratio,
        test_user_ratio=test_user_ratio,
        n_val_user=n_val_user,
        n_test_user=n_test_user,
        heldout_ratio_val=0.3,
        heldout_ratio_test=0.5,
        ceil_n_heldout=ceil,
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

    with pytest.raises(ValueError):
        _ = UserTrainTestInteractionPair(
            train.user_ids, train.X_train, train.X_train, mid_list[:-1]
        )

    def get_n_right_answer(ratio: float, n: Optional[int]) -> int:
        if n is not None:
            return n
        else:
            return int(n_users_all * ratio)

    val = dataset["val"]
    assert val.n_users == get_n_right_answer(val_user_ratio, n_val_user)
    test = dataset["test"]
    assert test.n_users == get_n_right_answer(test_user_ratio, n_test_user)

    if time_colname is not None:
        for d in [val, test]:
            _df_train = d.df_train().merge(
                df[["userId", "movieId", "timestamp"]].rename(
                    columns={"userId": "user_id", "movieId": "item_id"}
                )
            )
            _df_test = d.df_test().merge(
                df[["userId", "movieId", "timestamp"]].rename(
                    columns={"userId": "user_id", "movieId": "item_id"}
                )
            )
            _train_max_time = _df_train.groupby("user_id").timestamp.max()
            _test_min_time = _df_test.groupby("user_id").timestamp.min()
            common_index = np.intersect1d(_train_max_time.index, _test_min_time.index)
            assert common_index.shape[0] > 0
            assert np.all(
                _train_max_time.reindex(common_index)
                <= _test_min_time.reindex(common_index)
            )

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


def test_extreme_case() -> None:
    ratio = 0.3
    dataset, mid_list = split_dataframe_partial_user_holdout(
        df,
        user_column="userId",
        item_column="movieId",
        n_heldout_val=1,
        val_user_ratio=1.0,
        test_user_ratio=0,
        heldout_ratio_test=ratio,
    )
    assert len(mid_list) == len(set(df.movieId))

    assert dataset["train"].n_users == 0
    assert dataset["test"].n_users == 0
    assert dataset["val"].n_users == len(set(df.userId))
    assert dataset["val"].X_all.nnz == df.shape[0]
