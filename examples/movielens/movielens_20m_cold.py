import json
from typing import Any, Dict, List, Tuple, Type

import pandas as pd
from scipy import sparse as sps

from irspack import (
    AsymmetricCosineKNNRecommender,
    BaseRecommender,
    CosineKNNRecommender,
    DenseSLIMRecommender,
    EvaluatorWithColdUser,
    IALSRecommender,
    P3alphaRecommender,
    RP3betaRecommender,
    SLIMRecommender,
    TopPopRecommender,
    split_dataframe_partial_user_holdout,
)
from irspack.dataset import MovieLens20MDataManager

if __name__ == "__main__":

    BASE_CUTOFF = 100

    # We follow the preprocessing of Mult-VAE implementation (https://github.com/dawenl/vae_cf)
    data_manager = MovieLens20MDataManager()
    df_all = data_manager.read_interaction()
    df_all = df_all[df_all.rating >= 4]
    user_cnt = df_all.userId.value_counts()
    user_cnt = user_cnt[user_cnt >= 5]
    df_all = df_all[df_all.userId.isin(user_cnt.index)]

    data_all, _ = split_dataframe_partial_user_holdout(
        df_all,
        "userId",
        "movieId",
        n_test_user=10000,
        n_val_user=10000,
        heldout_ratio_val=0.2,
        heldout_ratio_test=0.2,
    )

    data_train = data_all["train"]
    data_val = data_all["val"]
    data_test = data_all["test"]

    X_train_val_all: sps.csr_matrix = sps.vstack(
        [data_train.X_all, data_val.X_all], format="csr"
    )
    valid_evaluator = EvaluatorWithColdUser(
        input_interaction=data_val.X_train,
        ground_truth=data_val.X_test,
        cutoff=BASE_CUTOFF,
    )
    test_evaluator = EvaluatorWithColdUser(
        input_interaction=data_test.X_train,
        ground_truth=data_test.X_test,
        cutoff=BASE_CUTOFF,
    )

    test_results = []
    validation_results = []

    test_configs: List[Tuple[Type[BaseRecommender], int, Dict[str, Any]]] = [
        (TopPopRecommender, 1, dict()),
        (CosineKNNRecommender, 40, dict()),
        (AsymmetricCosineKNNRecommender, 40, dict()),
        (P3alphaRecommender, 30, dict(alpha=1)),
        (RP3betaRecommender, 40, dict(alpha=1)),
        (DenseSLIMRecommender, 20, dict()),
        (SLIMRecommender, 40, dict()),  # Note: this is a heavy one.
    ]
    try:
        from irspack import MultVAERecommender

        test_configs.append(
            (
                MultVAERecommender,
                1,
                dict(
                    dim_z=200, enc_hidden_dims=600, kl_anneal_goal=0.2
                ),  # nothing to tune, use the parameters used in the paper.
            )
        )
    except ImportError:
        pass

    for recommender_class, n_trials, config in test_configs:
        recommender_name = recommender_class.__name__
        (best_param, validation_result_df) = recommender_class.tune(
            data_train.X_all,
            valid_evaluator,
            fixed_params=config,
            n_trials=n_trials,
            random_seed=0,
        )
        validation_result_df["recommender_name"] = recommender_name
        validation_results.append(validation_result_df)
        pd.concat(validation_results).to_csv(f"validation_scores.csv")
        test_recommender = recommender_class(X_train_val_all, **best_param).learn()
        test_scores = test_evaluator.get_scores(test_recommender, [20, 50, 100])

        test_results.append(
            dict(name=recommender_name, best_param=best_param, **test_scores)
        )
        with open("test_results.json", "w") as ofs:
            json.dump(test_results, ofs, indent=2)

    # Tuning following the strategy of
    # "Revisiting the Performance of iALS on Item Recommendation Benchmarks"
    # https://arxiv.org/abs/2110.14037
    (
        best_param_ials,
        validation_result_df_ials,
    ) = IALSRecommender.tune_doubling_dimension(
        data_train.X_all,
        valid_evaluator,
        initial_dimension=128,
        maximal_dimension=1024,
        random_seed=0,
        n_trials_initial=80,
        n_trials_following=40,
    )
    validation_result_df_ials["recommender_name"] = "IALSRecommender"
    validation_results.append(validation_result_df_ials)
    pd.concat(validation_results).to_csv(f"validation_scores.csv")
    test_recommender_ials = IALSRecommender(X_train_val_all, **best_param_ials)
    test_recommender_ials.learn()
    test_scores_ials = test_evaluator.get_scores(test_recommender_ials, [20, 50, 100])

    test_results.append(
        dict(name="IALSRecommender", best_param=best_param_ials, **test_scores_ials)
    )
    with open("test_results.json", "w") as ofs:
        json.dump(test_results, ofs, indent=2)
