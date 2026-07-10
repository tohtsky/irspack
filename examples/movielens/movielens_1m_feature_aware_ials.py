"""Tune iALS and compare feature-only cold-user recommendations on ML-1M."""

import json
from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna
import pandas as pd
from scipy import sparse as sps
from sklearn.preprocessing import MultiLabelBinarizer

from irspack.dataset.movielens import MovieLens1MDataManager
from irspack.evaluation import EvaluatorWithColdUser
from irspack.recommenders.ials import IALSRecommender
from irspack.recommenders.toppop import TopPopRecommender
from irspack.split import split_dataframe_partial_user_holdout


def make_item_features(item_info: pd.DataFrame, item_ids: list[Any]) -> sps.csr_matrix:
    ordered = item_info.reindex(item_ids)
    genre = MultiLabelBinarizer(sparse_output=True).fit_transform(
        ordered["genres"].fillna("").str.split("|")
    )
    year = ordered["release_year"].fillna(ordered["release_year"].median())
    year_scaled = ((year - year.mean()) / year.std()).to_numpy(dtype=np.float32)[
        :, None
    ]
    bias = np.ones((len(item_ids), 1), dtype=np.float32)
    return sps.hstack(
        [genre.astype(np.float32), sps.csr_matrix(year_scaled), bias],
        format="csr",
        dtype=np.float32,
    )


def make_user_features(user_info: pd.DataFrame, user_ids: list[Any]) -> sps.csr_matrix:
    ordered = user_info.reindex(user_ids)
    categorical_labels = [
        [
            f"gender={row.gender}",
            f"age={row.age}",
            f"occupation={row.occupation}",
            # Keep the dense ridge system small; feature weights currently use
            # a dense Gram matrix.
            f"zipcode_region={str(row.zipcode)[:1]}",
        ]
        for row in ordered.itertuples()
    ]
    categorical = MultiLabelBinarizer(sparse_output=True).fit_transform(
        categorical_labels
    )
    bias = np.ones((len(user_ids), 1), dtype=np.float32)
    return sps.hstack(
        [categorical.astype(np.float32), bias],
        format="csr",
        dtype=np.float32,
    )


def train_model(
    X: sps.csr_matrix,
    config: Dict[str, Any],
    user_features: Optional[sps.csr_matrix],
    item_features: Optional[sps.csr_matrix],
) -> IALSRecommender:
    feature_config: Dict[str, Any] = {}
    if user_features is not None or item_features is not None:
        feature_config = {
            "user_features": user_features,
            "item_features": item_features,
            "lambda_user_feature": config["lambda_user_feature"],
            "lambda_item_feature": config["lambda_item_feature"],
        }
    return IALSRecommender(
        X,
        n_components=config["n_components"],
        alpha0=config["alpha0"],
        reg=config["reg"],
        nu=1,
        solver_type="CG",
        max_cg_steps=3,
        loss_type="IALSPP",
        train_epochs=16,
        n_threads=4,
        random_seed=0,
        **feature_config,
    ).learn()


def train_and_score(
    X: sps.csr_matrix,
    evaluator: EvaluatorWithColdUser,
    config: Dict[str, Any],
    user_features: Optional[sps.csr_matrix],
    item_features: Optional[sps.csr_matrix],
) -> Dict[str, float]:
    model = train_model(X, config, user_features, item_features)
    return dict(evaluator.get_scores(model, [5, 10, 20]))


def feature_only_scores(
    model: IALSRecommender, user_features: sps.csr_matrix
) -> np.ndarray:
    return model.get_score_cold_user_from_features(user_features)


def suggest_common_config(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_components": 192,
        "alpha0": trial.suggest_float("alpha0", 0.03, 1.0, log=True),
        "reg": trial.suggest_float("reg", 1e-4, 1e-1, log=True),
    }


def make_validation_objective(
    model_type: str,
    X: sps.csr_matrix,
    validation_evaluator: EvaluatorWithColdUser,
    fully_cold_validation_evaluator: EvaluatorWithColdUser,
    train_user_features: sps.csr_matrix,
    validation_user_features: sps.csr_matrix,
    item_features: sps.csr_matrix,
    validation_rows: list[Dict[str, Any]],
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        config = suggest_common_config(trial)
        features = None
        user_features = None
        if model_type == "feature-aware":
            config["lambda_user_feature"] = trial.suggest_float(
                "lambda_user_feature", 5e-2, 1e6, log=True
            )
            config["lambda_item_feature"] = trial.suggest_float(
                "lambda_item_feature", 5e-2, 1e6, log=True
            )
            features = item_features
            user_features = train_user_features

        try:
            model = train_model(X, config, user_features, features)
        except RuntimeError as err:
            if "Feature ridge Cholesky decomposition failed" in str(err):
                raise optuna.TrialPruned(str(err)) from err
            raise
        scores = dict(validation_evaluator.get_scores(model, [5, 10, 20]))
        if features is not None:
            cold_scores = feature_only_scores(model, validation_user_features)
            fully_cold_scores = (
                fully_cold_validation_evaluator.get_scores_from_score_matrix(
                    cold_scores, [5, 10, 20]
                )
            )
            scores.update(
                {
                    f"fully_cold_{metric}": value
                    for metric, value in fully_cold_scores.items()
                }
            )
            objective_score = scores["ndcg@20"]
        else:
            objective_score = scores["ndcg@20"]

        validation_rows.append(
            {
                "model_type": model_type,
                "trial_number": trial.number,
                "objective_score": objective_score,
                **config,
                **scores,
            }
        )
        return float(objective_score)

    return objective


if __name__ == "__main__":
    manager = MovieLens1MDataManager()
    interactions = manager.read_interaction()
    split, item_ids = split_dataframe_partial_user_holdout(
        interactions,
        "userId",
        "movieId",
        test_user_ratio=0.2,
        val_user_ratio=0.2,
        heldout_ratio_test=0.5,
        heldout_ratio_val=0.5,
        random_state=0,
    )
    train, validation, test = split["train"], split["val"], split["test"]
    item_features = make_item_features(manager.read_item_info(), item_ids)
    user_info = manager.read_user_info()
    all_split_user_ids = train.user_ids + validation.user_ids + test.user_ids
    all_user_features = make_user_features(user_info, all_split_user_ids)
    train_end = train.n_users
    validation_end = train_end + validation.n_users
    train_user_features = all_user_features[:train_end]
    validation_user_features = all_user_features[train_end:validation_end]
    test_user_features = all_user_features[validation_end:]
    validation_evaluator = EvaluatorWithColdUser(
        validation.X_train, validation.X_test, cutoff=20, n_threads=4
    )
    validation_no_interactions = sps.csr_matrix(
        validation.X_all.shape, dtype=np.float32
    )
    fully_cold_validation_evaluator = EvaluatorWithColdUser(
        validation_no_interactions,
        validation.X_all,
        cutoff=20,
        n_threads=4,
        mb_size=validation.n_users,
    )

    validation_rows: list[Dict[str, Any]] = []
    baseline_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        study_name="movielens-1m-baseline-ials",
    )
    baseline_study.optimize(
        make_validation_objective(
            "baseline",
            train.X_all,
            validation_evaluator,
            fully_cold_validation_evaluator,
            train_user_features,
            validation_user_features,
            item_features,
            validation_rows,
        ),
        n_trials=30,
    )
    feature_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=1),
        study_name="movielens-1m-feature-aware-ials",
    )
    feature_study.optimize(
        make_validation_objective(
            "feature-aware",
            train.X_all,
            validation_evaluator,
            fully_cold_validation_evaluator,
            train_user_features,
            validation_user_features,
            item_features,
            validation_rows,
        ),
        n_trials=120,
    )
    validation_results = pd.DataFrame(validation_rows)
    validation_results.to_csv("feature_aware_ials_validation.csv", index=False)

    best_baseline = validation_results.loc[
        validation_results["model_type"] == "baseline", "ndcg@20"
    ].idxmax()
    best_feature = validation_results.loc[
        validation_results["model_type"] == "feature-aware",
        "objective_score",
    ].idxmax()
    best_rows = validation_results.loc[[best_baseline, best_feature]]
    X_train_validation = sps.vstack([train.X_all, validation.X_all], format="csr")
    train_validation_user_features = sps.vstack(
        [train_user_features, validation_user_features], format="csr"
    )
    test_evaluator = EvaluatorWithColdUser(
        test.X_train, test.X_test, cutoff=20, n_threads=4
    )
    test_results = []
    for _, row in best_rows.iterrows():
        model_type = str(row["model_type"])
        config = {
            "n_components": int(row["n_components"]),
            "alpha0": float(row["alpha0"]),
            "reg": float(row["reg"]),
        }
        features = None
        if model_type == "feature-aware":
            config["lambda_user_feature"] = float(row["lambda_user_feature"])
            config["lambda_item_feature"] = float(row["lambda_item_feature"])
            features = item_features
        user_features = train_validation_user_features if features is not None else None
        scores = train_and_score(
            X_train_validation,
            test_evaluator,
            config,
            user_features,
            features,
        )
        test_results.append(
            {"model_type": model_type, "config": config, "scores": scores}
        )
    with open("feature_aware_ials_test.json", "w") as output:
        json.dump(test_results, output, indent=2)
    print(json.dumps(test_results, indent=2))

    # Hold out every interaction of the test users. Their iALS embeddings are
    # obtained only from demographics; no interaction history is provided.
    feature_row = best_rows.loc[best_rows["model_type"] == "feature-aware"].iloc[0]
    feature_config = {
        "n_components": int(feature_row["n_components"]),
        "alpha0": float(feature_row["alpha0"]),
        "reg": float(feature_row["reg"]),
        "lambda_user_feature": float(feature_row["lambda_user_feature"]),
        "lambda_item_feature": float(feature_row["lambda_item_feature"]),
    }
    feature_model = train_model(
        X_train_validation,
        feature_config,
        train_validation_user_features,
        item_features,
    )
    test_feature_only_scores = feature_only_scores(feature_model, test_user_features)

    no_interactions = sps.csr_matrix(test.X_all.shape, dtype=np.float32)
    fully_cold_evaluator = EvaluatorWithColdUser(
        no_interactions,
        test.X_all,
        cutoff=20,
        n_threads=4,
        mb_size=test.n_users,
    )
    top_pop = TopPopRecommender(X_train_validation).learn()
    top_pop_scores = dict(fully_cold_evaluator.get_scores(top_pop, [5, 10, 20]))
    feature_only_metrics = dict(
        fully_cold_evaluator.get_scores_from_score_matrix(
            test_feature_only_scores, [5, 10, 20]
        )
    )
    fully_cold_results = {
        "n_fully_held_out_users": test.n_users,
        "top_pop": top_pop_scores,
        "feature_aware_ials": feature_only_metrics,
        "feature_aware_minus_top_pop": {
            metric: feature_only_metrics[metric] - value
            for metric, value in top_pop_scores.items()
        },
    }
    with open("feature_aware_ials_fully_cold_test.json", "w") as output:
        json.dump(fully_cold_results, output, indent=2)
    print(json.dumps(fully_cold_results, indent=2))
