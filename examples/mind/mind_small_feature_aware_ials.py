"""Compare ordinary and feature-aware iALS on a temporal MIND-small split.

The final day of the official train archive is used for validation, and the
official dev archive is used for testing.  Recommendation candidates are the
articles exposed during each evaluation period.  Only users with pre-cutoff
history are evaluated here.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple

import numpy as np
import optuna
import pandas as pd
from scipy import sparse as sps
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, OneHotEncoder

from irspack import EvaluatorWithColdUser, IALSRecommender, df_to_sparse
from irspack.dataset import MINDDataManager

TRAIN_EPOCHS = 16
EVALUATION_CUTOFF = 20
METRICS_TO_REPORT = (
    "ndcg@20",
    "recall@20",
    "hit@20",
    "gini_index@20",
    "entropy@20",
    "catalog_coverage@20",
)


@dataclass
class TemporalPeriod:
    X_train: sps.csr_matrix
    train_item_popularity: np.ndarray
    eval_item_ids: np.ndarray
    item_features_train: sps.csr_matrix
    item_features_eval: sps.csr_matrix
    warm_item_indices: np.ndarray
    cold_item_indices: np.ndarray
    recommendable_item_indices: np.ndarray
    evaluators: Dict[str, EvaluatorWithColdUser]


def make_item_features(
    item_info: pd.DataFrame,
    fit_item_ids: np.ndarray,
    output_item_ids: np.ndarray,
) -> sps.csr_matrix:
    """Fit title TF-IDF/SVD and category transforms on pre-cutoff items."""
    fit_info = item_info.reindex(fit_item_ids)
    output_info = item_info.reindex(output_item_ids)
    vectorizer = TfidfVectorizer(
        max_features=30_000,
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english",
        dtype=np.float32,
    )
    fit_text = vectorizer.fit_transform(fit_info["title"].fillna(""))
    n_components = min(128, fit_text.shape[0] - 1, fit_text.shape[1] - 1)
    if n_components < 1:
        raise RuntimeError("Not enough pre-cutoff item text to fit item features.")

    svd = TruncatedSVD(n_components=n_components, random_state=0)
    svd.fit(fit_text)
    text_features = sps.csr_matrix(
        Normalizer(copy=False)
        .fit_transform(
            svd.transform(vectorizer.transform(output_info["title"].fillna("")))
        )
        .astype(np.float32)
    )
    categorical_columns = ["category", "subcategory"]
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=True, dtype=np.float32
    )
    encoder.fit(fit_info[categorical_columns].fillna("__missing__"))
    categorical = encoder.transform(
        output_info[categorical_columns].fillna("__missing__")
    )
    bias = sps.csr_matrix(np.ones((len(output_item_ids), 1), dtype=np.float32))
    return sps.hstack(
        [text_features, categorical, bias], format="csr", dtype=np.float32
    )


def build_period(
    train_events: pd.DataFrame,
    history_events: pd.DataFrame,
    target_events: pd.DataFrame,
    item_info: pd.DataFrame,
    candidate_item_ids: np.ndarray,
) -> TemporalPeriod:
    """Build matrices using the evaluation period's exposed-item catalog."""
    train_item_ids = np.sort(train_events["item_id"].unique())
    train_item_popularity = (
        train_events["item_id"]
        .value_counts()
        .reindex(train_item_ids, fill_value=0)
        .to_numpy(dtype=np.float32)
    )
    metadata_item_ids = item_info.index.to_numpy()
    if np.setdiff1d(train_item_ids, metadata_item_ids).size:
        raise RuntimeError("Some training items have no MIND article metadata.")

    candidate_item_ids = np.unique(candidate_item_ids)
    if np.setdiff1d(candidate_item_ids, metadata_item_ids).size:
        raise RuntimeError("Some candidate items have no MIND article metadata.")
    if np.setdiff1d(target_events["item_id"].unique(), candidate_item_ids).size:
        raise RuntimeError("Some target items are outside the candidate catalog.")

    eval_item_ids = np.union1d(train_item_ids, candidate_item_ids)
    X_train, _, _ = df_to_sparse(
        train_events, "user_id", "item_id", item_ids=train_item_ids
    )
    X_train = X_train.tocsr().astype(np.float32)
    X_train.data[:] = 1

    target_user_ids = np.sort(target_events["user_id"].unique())
    X_history, _, _ = df_to_sparse(
        history_events,
        "user_id",
        "item_id",
        user_ids=target_user_ids,
        item_ids=eval_item_ids,
    )
    X_target, _, _ = df_to_sparse(
        target_events,
        "user_id",
        "item_id",
        user_ids=target_user_ids,
        item_ids=eval_item_ids,
    )
    X_history = X_history.tocsr().astype(np.float32)
    X_target = X_target.tocsr().astype(np.float32)
    X_history.data[:] = 1
    X_target.data[:] = 1

    eval_item_index = pd.Series(np.arange(len(eval_item_ids)), index=eval_item_ids)
    warm_item_indices = eval_item_index.loc[train_item_ids].to_numpy(dtype=np.int64)
    recommendable_item_indices = eval_item_index.loc[candidate_item_ids].to_numpy(
        dtype=np.int64
    )
    has_history = np.asarray(X_history[:, warm_item_indices].getnnz(axis=1) > 0)
    X_history = X_history[has_history]
    X_target = X_target[has_history]
    if X_history.shape[0] == 0:
        raise RuntimeError("The evaluation period has no users with history.")

    cold_target = X_target.copy().tolil()
    cold_target[:, warm_item_indices] = 0
    cold_target = cold_target.tocsr()
    has_cold_target = np.asarray(cold_target.getnnz(axis=1) > 0)
    if not has_cold_target.any():
        raise RuntimeError("No users with history have a new-item target.")

    evaluator_args = {
        "cutoff": EVALUATION_CUTOFF,
        "recommendable_items": recommendable_item_indices.tolist(),
        "mb_size": 256,
    }
    cold_item_indices = np.setdiff1d(np.arange(len(eval_item_ids)), warm_item_indices)
    warm_target = X_target.copy().tolil()
    warm_target[:, cold_item_indices] = 0
    warm_target = warm_target.tocsr()
    has_warm_target = np.asarray(warm_target.getnnz(axis=1) > 0)
    evaluators = {
        "all_targets": EvaluatorWithColdUser(X_history, X_target, **evaluator_args),
        "new_item_targets": EvaluatorWithColdUser(
            X_history[has_cold_target],
            cold_target[has_cold_target],
            **evaluator_args,
        ),
        "warm_item_targets": EvaluatorWithColdUser(
            X_history[has_warm_target],
            warm_target[has_warm_target],
            **evaluator_args,
        ),
    }
    item_features_eval = make_item_features(item_info, train_item_ids, eval_item_ids)
    return TemporalPeriod(
        X_train=X_train,
        train_item_popularity=train_item_popularity,
        eval_item_ids=eval_item_ids,
        item_features_train=item_features_eval[warm_item_indices],
        item_features_eval=item_features_eval,
        warm_item_indices=warm_item_indices,
        cold_item_indices=cold_item_indices,
        recommendable_item_indices=recommendable_item_indices,
        evaluators=evaluators,
    )


def train_model(
    period: TemporalPeriod, config: Dict[str, Any], feature_aware: bool
) -> IALSRecommender:
    feature_args: Dict[str, Any] = {}
    if feature_aware:
        feature_args = {
            "item_features": period.item_features_train,
            "lambda_item_feature": config["lambda_item_feature"],
        }
    return IALSRecommender(
        period.X_train,
        n_components=config["n_components"],
        alpha0=config["alpha0"],
        reg=config["reg"],
        train_epochs=int(config.get("train_epochs", TRAIN_EPOCHS)),
        random_seed=0,
        **feature_args,
    ).learn()


def _metrics_from_scores(
    evaluator: EvaluatorWithColdUser,
    score_chunks: Iterable[np.ndarray],
    n_recommendable: int,
) -> Dict[str, float]:
    scores = evaluator.get_scores_from_score_chunks(score_chunks, [EVALUATION_CUTOFF])
    scores["catalog_coverage@20"] = scores["appeared_item@20"] / n_recommendable
    return {name: scores[name] for name in METRICS_TO_REPORT}


def score_model(
    model: IALSRecommender,
    period: TemporalPeriod,
    evaluator: EvaluatorWithColdUser,
    feature_aware: bool,
) -> Dict[str, float]:
    """Evaluate in batches without materializing the full MIND score matrix."""

    def chunks() -> Iterator[np.ndarray]:
        cold_embedding = None
        if feature_aware and period.cold_item_indices.size:
            cold_embedding = model.compute_item_embedding_from_features(
                period.item_features_eval[period.cold_item_indices]
            )
        for begin in range(0, evaluator.n_users, evaluator.mb_size):
            end = min(begin + evaluator.mb_size, evaluator.n_users)
            history = evaluator.input_interaction[begin:end]
            user_embedding = model.compute_user_embedding(
                history[:, period.warm_item_indices]
            )
            scores = np.full(
                (end - begin, len(period.eval_item_ids)),
                -np.inf,
                dtype=user_embedding.dtype,
            )
            scores[:, period.warm_item_indices] = model.get_score_from_user_embedding(
                user_embedding
            )
            if cold_embedding is not None:
                scores[:, period.cold_item_indices] = user_embedding.dot(
                    cold_embedding.T
                )
            yield scores

    return _metrics_from_scores(
        evaluator, chunks(), len(period.recommendable_item_indices)
    )


def score_top_pop(
    period: TemporalPeriod, evaluator: EvaluatorWithColdUser
) -> Dict[str, float]:
    popularity = np.full(len(period.eval_item_ids), -np.inf, dtype=np.float32)
    popularity[period.warm_item_indices] = period.train_item_popularity

    def chunks() -> Iterator[np.ndarray]:
        for begin in range(0, evaluator.n_users, evaluator.mb_size):
            end = min(begin + evaluator.mb_size, evaluator.n_users)
            yield np.repeat(popularity[None, :], end - begin, axis=0)

    return _metrics_from_scores(
        evaluator, chunks(), len(period.recommendable_item_indices)
    )


class TemporalTuningEvaluator(EvaluatorWithColdUser):
    """Connect period-catalog scoring to ``IALSRecommender.tune``."""

    def __init__(self, period: TemporalPeriod, feature_aware: bool):
        self.period = period
        self.feature_aware = feature_aware
        evaluator = period.evaluators["all_targets"]
        self.target_metric = evaluator.target_metric
        self.cutoff = evaluator.cutoff
        self.target_metric_name = evaluator.target_metric_name

    def get_score(self, model: IALSRecommender) -> Dict[str, float]:
        scores = score_model(
            model,
            self.period,
            self.period.evaluators["all_targets"],
            self.feature_aware,
        )
        return {name.split("@", 1)[0]: value for name, value in scores.items()}


def tune(
    period: TemporalPeriod,
    feature_aware: bool,
    n_trials: int,
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    def suggest_parameters(trial: optuna.Trial) -> Dict[str, Any]:
        parameters: Dict[str, Any] = {
            "n_components": trial.suggest_int("n_components", 32, 256, log=True),
            "alpha0": trial.suggest_float("alpha0", 0.03, 1.0, log=True),
            "reg": trial.suggest_float("reg", 1e-4, 1e-1, log=True),
        }
        if feature_aware:
            parameters["lambda_item_feature"] = trial.suggest_float(
                "lambda_item_feature", 1.0, 1e6, log=True
            )
        return parameters

    fixed_parameters: Dict[str, Any] = {
        "random_seed": 0,
    }
    if feature_aware:
        fixed_parameters["item_features"] = period.item_features_train
    return IALSRecommender.tune(
        period.X_train,
        TemporalTuningEvaluator(period, feature_aware),
        n_trials=n_trials,
        tuning_random_seed=seed,
        parameter_suggest_function=suggest_parameters,
        validate_epoch=1,
        score_degradation_max=3,
        **fixed_parameters,
    )


def evaluate_period(
    period: TemporalPeriod,
    baseline_config: Dict[str, Any],
    feature_config: Dict[str, Any],
    segment_names: Tuple[str, ...] = ("all_targets", "new_item_targets"),
    include_top_pop: bool = False,
) -> Dict[str, Any]:
    models = {
        "iALS": (
            train_model(period, baseline_config, False),
            False,
        ),
        "feature-aware iALS": (
            train_model(period, feature_config, True),
            True,
        ),
    }
    segments: Dict[str, Any] = {}
    for segment_name in segment_names:
        evaluator = period.evaluators[segment_name]
        scores_by_model = {
            model_name: score_model(model, period, evaluator, feature_aware)
            for model_name, (model, feature_aware) in models.items()
        }
        if include_top_pop:
            scores_by_model["TopPop"] = score_top_pop(period, evaluator)
        segments[segment_name] = {
            metric: {
                model_name: model_scores[metric]
                for model_name, model_scores in scores_by_model.items()
            }
            for metric in METRICS_TO_REPORT
        }
    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("mind_feature_aware_ials_test.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with MINDDataManager(force_download=args.force_download) as manager:
        official_train = manager.read_interaction("train")
        official_dev = manager.read_interaction("dev")
        train_impressions = manager.read_impressions("train")
        dev_impressions = manager.read_impressions("dev")
        train_item_info = manager.read_item_info("train")
        all_item_info = manager.read_item_info()

    validation_start = official_train["timestamp"].max().normalize()
    validation_train = official_train[official_train["timestamp"] < validation_start]
    validation_target = official_train[official_train["timestamp"] >= validation_start]
    validation_period = build_period(
        validation_train,
        validation_train,
        validation_target,
        train_item_info,
        candidate_item_ids=train_impressions.loc[
            train_impressions["timestamp"] >= validation_start, "item_id"
        ].unique(),
    )

    baseline_config, baseline_history = tune(
        validation_period, False, args.n_trials, seed=0
    )
    feature_config, feature_history = tune(
        validation_period, True, args.n_trials, seed=1
    )
    baseline_history.insert(0, "model", "iALS")
    feature_history.insert(0, "model", "feature-aware iALS")
    pd.concat([baseline_history, feature_history]).to_csv(
        args.output.with_name(f"{args.output.stem}_validation.csv"), index=False
    )

    test_period = build_period(
        official_train,
        official_train,
        official_dev,
        all_item_info,
        candidate_item_ids=dev_impressions["item_id"].unique(),
    )
    results = {
        "validation_start": validation_start.isoformat(),
        "model_config": {
            "iALS": baseline_config,
            "feature-aware iALS": feature_config,
        },
        "validation": evaluate_period(
            validation_period, baseline_config, feature_config
        ),
        "test": evaluate_period(
            test_period,
            baseline_config,
            feature_config,
            segment_names=(
                "all_targets",
                "warm_item_targets",
                "new_item_targets",
            ),
            include_top_pop=True,
        ),
    }
    with args.output.open("w") as output:
        json.dump(results, output, indent=2)
    print(json.dumps(results, indent=2))
