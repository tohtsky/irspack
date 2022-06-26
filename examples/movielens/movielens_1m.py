import json
from typing import List, Tuple, Type

from scipy import sparse as sps

from irspack import (  # BPRFMRecommender, #requires lightFM; MultVAERecommender, #requires jax & haiku & optax
    AsymmetricCosineKNNRecommender,
    BaseRecommender,
    CosineKNNRecommender,
    DenseSLIMRecommender,
    Evaluator,
    IALSRecommender,
    P3alphaRecommender,
    RP3betaRecommender,
    SLIMRecommender,
    TopPopRecommender,
    TverskyIndexKNNRecommender,
)
from irspack.dataset import MovieLens1MDataManager
from irspack.split import split_dataframe_partial_user_holdout

if __name__ == "__main__":

    BASE_CUTOFF = 20

    data_manager = MovieLens1MDataManager()
    df_all = data_manager.read_interaction()

    data_all, _ = split_dataframe_partial_user_holdout(
        df_all,
        "userId",
        "movieId",
        test_user_ratio=0.2,
        val_user_ratio=0.2,
        heldout_ratio_test=0.5,
        heldout_ratio_val=0.5,
    )

    data_train = data_all["train"]
    data_val = data_all["val"]
    data_test = data_all["test"]

    X_train_all: sps.csr_matrix = sps.vstack(
        [data_train.X_train, data_val.X_train, data_test.X_train], format="csr"
    )
    X_train_val_all: sps.csr_matrix = sps.vstack(
        [data_train.X_all, data_val.X_all, data_test.X_train], format="csr"
    )
    valid_evaluator = Evaluator(
        ground_truth=data_val.X_test,
        offset=data_train.n_users,
        cutoff=BASE_CUTOFF,
    )
    test_evaluator = Evaluator(
        ground_truth=data_test.X_test,
        offset=data_train.n_users + data_val.n_users,
        cutoff=BASE_CUTOFF,
    )

    test_results = []

    test_configs: List[Tuple[Type[BaseRecommender], int]] = [
        (TopPopRecommender, 1),
        (CosineKNNRecommender, 40),
        (AsymmetricCosineKNNRecommender, 40),
        (TverskyIndexKNNRecommender, 40),
        (DenseSLIMRecommender, 20),
        (P3alphaRecommender, 40),
        (RP3betaRecommender, 40),
        (IALSRecommender, 40),
        (SLIMRecommender, 40),
        # (BPRFMRecommender, 40),
        # (MultVAERecommender, 5),
    ]
    for recommender_class, n_trials in test_configs:
        name = recommender_class.__name__
        (best_param, validation_results) = recommender_class.tune(
            X_train_all,
            valid_evaluator,
            timeout=14400,
            n_trials=n_trials,
            random_seed=0,
        )
        validation_results.to_csv(f"{name}_validation_scores.csv")
        test_recommender = recommender_class(X_train_val_all, **best_param).learn()
        test_scores = test_evaluator.get_scores(test_recommender, [5, 10, 20])

        test_results.append(dict(name=name, best_param=best_param, **test_scores))
        with open("test_results.json", "w") as ofs:
            json.dump(test_results, ofs, indent=2)
