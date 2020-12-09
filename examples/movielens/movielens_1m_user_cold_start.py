from typing import Type, List, Tuple
from scipy import sparse as sps
import numpy as np
from irspack.dataset.movielens import MovieLens1MDataManager
from irspack.utils.encoders import (
    CategoricalValueEncoder,
    DataFrameEncoder,
    BinningEncoder,
)

from irspack.user_cold_start import (
    UserColdStartRecommenderBase,
    UserColdStartEvaluator,
    UserCBKNNRecommender,
    LinearRecommender,
    TopPopularRecommender,
    CB2IALSOptimizer,
    CB2TruncatedSVDOptimizer,
    CB2BPRFMOptimizer,
)

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    loader = MovieLens1MDataManager()
    user_df = loader.read_user_info()
    ratings = loader.load_rating()
    user_ids_unique = np.unique(ratings.userId)
    movie_ids_unique = np.unique(ratings.movieId)
    movie_id_to_index = {id_: i for i, id_ in enumerate(movie_ids_unique)}

    def df_to_sparse(df):
        unique_uids, row = np.unique(df.userId, return_inverse=True)
        col = df.movieId.map(movie_id_to_index)
        return (
            unique_uids,
            sps.csr_matrix(
                (np.ones(df.shape[0]), (row, col)),
                shape=(len(unique_uids), len(movie_id_to_index)),
            ),
        )

    train_uids, test_uids = train_test_split(
        user_ids_unique, test_size=0.2, random_state=42
    )
    train_uids, X_train = df_to_sparse(ratings[ratings.userId.isin(train_uids)])
    test_uids, X_test = df_to_sparse(ratings[ratings.userId.isin(test_uids)])

    ### Preprocess use data
    user_df["zip_first"] = user_df.zipcode.str[0]

    columns = ["occupation", "zip_first"]
    encoder_all = DataFrameEncoder()
    encoder_all.add_column(
        "gender", CategoricalValueEncoder(user_df["gender"])
    ).add_column("age", BinningEncoder(user_df["age"], n_percentiles=10)).add_column(
        "occupation", CategoricalValueEncoder(user_df["occupation"])
    ).add_column(
        "zip_first", CategoricalValueEncoder(user_df["zip_first"])
    )

    user_info_train = encoder_all.transform_sparse(user_df.reindex(train_uids))
    user_info_test = encoder_all.transform_sparse(user_df.reindex(test_uids))

    test_evaluator = UserColdStartEvaluator(X_test, user_info_test)

    trial_configurations: List[Tuple[Type[UserColdStartRecommenderBase], int]] = [
        (TopPopularRecommender, 1),
        (UserCBKNNRecommender, 20),
        (LinearRecommender, 10),
    ]
    for recommender_class, n_trial in trial_configurations:
        best_param = recommender_class.optimize(
            X_train, user_info_train, n_trials=n_trial
        )
        rec = recommender_class(X_train, user_info_train, **best_param)
        rec.learn()
        print(test_evaluator.get_score(rec, 20))

    for cboptim_class in [CB2TruncatedSVDOptimizer]:
        optimizer = cboptim_class(X_train, user_info_train)
        rec, best_cf_config, best_nn_config = optimizer.search_all(n_trials=20)
        print(test_evaluator.get_score(rec))
