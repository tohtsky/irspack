import json
import logging
import os
from typing import List, Tuple, Type

from scipy import sparse as sps

from irspack.dataset.movielens import MovieLens1MDataManager
from irspack.evaluator import Evaluator
from irspack.optimizers import (  # BPRFMOptimizer, #requires lightFM; MultVAEOptimizer, #requires jax & haiku & optax; SLIMOptimizer,
    AsymmetricCosineKNNOptimizer,
    BaseOptimizer,
    CosineKNNOptimizer,
    DenseSLIMOptimizer,
    IALSOptimizer,
    P3alphaOptimizer,
    RP3betaOptimizer,
    TopPopOptimizer,
    TverskyIndexKNNOptimizer,
)
from irspack.split import split_dataframe_partial_user_holdout

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["IRSPACK_NUM_THREADS_DEFAULT"] = "8"

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

    test_configs: List[Tuple[Type[BaseOptimizer], int]] = [
        (TopPopOptimizer, 1),
        (CosineKNNOptimizer, 40),
        (AsymmetricCosineKNNOptimizer, 40),
        (TverskyIndexKNNOptimizer, 40),
        (DenseSLIMOptimizer, 20),
        (P3alphaOptimizer, 40),
        (RP3betaOptimizer, 40),
        (IALSOptimizer, 40),
        # (BPRFMOptimizer, 40),
        # (MultVAEOptimizer, 5),
        # (SLIMOptimizer, 40),
    ]
    for optimizer_class, n_trials in test_configs:
        name = optimizer_class.__name__
        optimizer: BaseOptimizer = optimizer_class(X_train_all, valid_evaluator)
        (best_param, validation_results) = optimizer.optimize(
            timeout=14400, n_trials=n_trials
        )
        validation_results.to_csv(f"{name}_validation_scores.csv")
        test_recommender = optimizer.recommender_class(X_train_val_all, **best_param)
        test_recommender.learn()
        test_scores = test_evaluator.get_scores(test_recommender, [5, 10, 20])

        test_results.append(dict(name=name, best_param=best_param, **test_scores))
        with open("test_results.json", "w") as ofs:
            json.dump(test_results, ofs, indent=2)
