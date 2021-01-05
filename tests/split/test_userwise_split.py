import numpy as np

from irspack.dataset.movielens import MovieLens100KDataManager
from irspack.split import split_dataframe_partial_user_holdout

RNS = np.random.RandomState(0)

df = MovieLens100KDataManager(force_download=True).read_interaction()


def test_user_level_split() -> None:
    dataset, mid_list = split_dataframe_partial_user_holdout(
        df,
        user_column="userId",
        item_column="movieId",
        n_test_user=30,
        n_val_user=30,
        heldout_ratio_test=0.5,
        heldout_ratio_val=0.3,
    )
    train = dataset["train"]
    assert train.X_test is None
    for key, ratio in [("val", 0.3), ("test", 0.5)]:
        train_predict_pair = dataset[key]
        X_learn = train_predict_pair.X_train
        X_predict = train_predict_pair.X_test
        intersect = X_learn.multiply(X_predict)
        assert intersect.count_nonzero() == 0
        index = RNS.choice(np.arange(train_predict_pair.n_users), size=10)
        for i in index:
            nnz_learn = X_learn[i].nonzero()[1].shape[0]
            nnz_predict = X_predict[i].nonzero()[1].shape[0]
            assert ratio > (nnz_predict - 1) / (nnz_learn + nnz_predict)
            assert ratio < (nnz_predict + 1) / (nnz_learn + nnz_predict)
